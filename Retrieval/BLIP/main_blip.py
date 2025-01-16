import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
import sys
import matplotlib.pyplot as plt
from bert_score.utils import get_idf_dict
import transformers
import argparse
import numpy as np
sys.path.append('/data/home/wangyouze/MultimodalAttack/VLP-attack-p/IT_Retrieval/')
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset.caption_dataset_bak import pair_dataset
from transformers import BertForMaskedLM
from PIL import Image
from torchvision import transforms
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from VLP_attack_BLIP import MultiModalAttacker
from tqdm import tqdm
sys.path.append('/data/home/wangyouze/projects/VLP_attack/models/blip')
from blip_retrieval import blip_retrieval
import utils 

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from jiwer import wer

sim_model = SentenceTransformer("/data/home/wangyouze/projects/others/multimodal_machine_translation/declutr-small/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_args(args):
	args_dict = vars(args)
	for arg_name, arg_value in sorted(args_dict.items()):
		print(f"\t{arg_name}: {arg_value}")

@torch.no_grad()
def retrieval_score(model, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text, device=None):
	if device is None:
		device = image_embeds.device

	sims_matrix = image_feats @ text_feats.t()
	score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

	for i, sims in enumerate(sims_matrix):
		topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

		encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
		encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
		output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
									attention_mask=text_atts[topk_idx].to(device),
									encoder_hidden_states=encoder_output,
									encoder_attention_mask=encoder_att,
									return_dict=True,
									mode='fusion'
									)
		score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
		score_matrix_i2t[i, topk_idx] = score

	sims_matrix = sims_matrix.t()
	score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

	for i, sims in enumerate(sims_matrix):
		topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
		encoder_output = image_embeds[topk_idx].to(device)
		encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
		output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
									attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
									encoder_hidden_states=encoder_output,
									encoder_attention_mask=encoder_att,
									return_dict=True,
									mode='fusion'
									)
		score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
		score_matrix_t2i[i, topk_idx] = score

	return score_matrix_i2t, score_matrix_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img):
	# Images->Text
	ranks = np.zeros(scores_i2t.shape[0])
	for index, score in enumerate(scores_i2t):
		inds = np.argsort(score)[::-1]
		# Score
		rank = 1e20
		for i in img2txt[index]:
			tmp = np.where(inds == i)[0][0]
			if tmp < rank:
				rank = tmp
		ranks[index] = rank

	# Compute metrics
	tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

	# Text->Images
	ranks = np.zeros(scores_t2i.shape[0])

	for index, score in enumerate(scores_t2i):
		inds = np.argsort(score)[::-1]
		ranks[index] = np.where(inds == txt2img[index])[0][0]

	# Compute metrics
	ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

	tr_mean = (tr1 + tr5 + tr10) / 3
	ir_mean = (ir1 + ir5 + ir10) / 3
	r_mean = (tr_mean + ir_mean) / 2

	eval_result = {'txt_r1': tr1,
				   'txt_r5': tr5,
				   'txt_r10': tr10,
				   'txt_r_mean': tr_mean,
				   'img_r1': ir1,
				   'img_r5': ir5,
				   'img_r10': ir10,
				   'img_r_mean': ir_mean,
				   'r_mean': r_mean}
	return eval_result

def calculate_similarity(origin_texts, adv_texts):
	token_errors = []
	res = []
	for i in range(len(origin_texts)):
		sentences = [origin_texts[i], adv_texts[i]]
		embeddings = sim_model.encode(sentences)
		token_errors.append(wer(sentences[0], sentences[1]))

		# Compute a semantic similarity via the cosine distance
		semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
		res.append(semantic_sim)
	return sum(res)/len(res)

def main(args, config):

	tokenizer = BertTokenizer.from_pretrained("/data/home/wangyouze/MultimodalAttack/bert-base-uncased/")
	tokenizer.model_max_length = 30

	#### Dataset ####
	print("Creating dataset")
	test_transform = transforms.Compose([
		transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
		transforms.ToTensor(),
	])
	test_dataset = pair_dataset(config['test_file'], test_transform, config['image_root'])

	test_loader = DataLoader(test_dataset, batch_size=30, num_workers=4)

	tokenizer = BertTokenizer.from_pretrained(args.text_encoder)


	#### Model ####
	print("Creating model")
	image_size = 384
	model_url = "/data/home/wangyouze/projects/checkpoints/model_base_retrieval_flickr.pth"
	model = blip_retrieval(med_config="/data/home/wangyouze/projects/VLP_attack/Retrieval/BLIP/configs/med_config.json", pretrained=model_url,
					 image_size=image_size, vit='base')
	
	model_ad = blip_retrieval(med_config="/data/home/wangyouze/projects/VLP_attack/Retrieval/BLIP/configs/med_config.json", pretrained=model_url,
					 image_size=image_size, vit='base')
	
	ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)
	model.eval()
	model = model.to(device)
	model_ad.eval()
	model_ad = model_ad.to(device)
	ref_model = ref_model.to(device)

   
	# visual_encoder_state_dict = torch.load("/data/home/wangyouze/projects/save/adversarial_robustness/TCL/TCL_TeCoA_eps_2_lr_1e_4/ViT-B_16_openai_imagenet_l2_imagenet_1NdKw/checkpoints/step_10.pt", map_location='cuda')
	# model_ad.visual_encoder.load_state_dict(visual_encoder_state_dict, strict=False)

	visual_encoder_1 = model.visual_encoder
	visual_encoder_2 = model.visual_encoder
  
	# model.eval()
	# model_ad.eval()

   
	# Compute idf dictionary for BERTScore
	idf_dict = get_idf_dict(test_dataset.text, tokenizer, nthreads=20)
	multimodal_attack = MultiModalAttacker(model, visual_encoder_1, visual_encoder_2, ref_model, idf_dict, tokenizer, end_idx=5000, transformation_num=7, device=device)
	images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

	num_text = len(test_loader.dataset.text)
	print('num_text:', num_text)
	num_image = len(test_loader.dataset.ann)
	print('num_image:', num_image)

	metric_logger = utils.MetricLogger(delimiter="  ")
	header = 'Evaluation:'    

	text_ids = []
	text_embeds = []  
	text_atts = []
	
	num_text = len(test_loader.dataset.text)
	num_image = len(test_loader.dataset.ann)
	image_embeds = torch.zeros(1000, 256)
	image_feats = torch.zeros(num_image, 577, 768)

	with torch.no_grad():
		tokens_id = torch.arange(0, tokenizer.vocab_size).long().cuda()
		embeddings = model.text_encoder.embeddings.word_embeddings(tokens_id).to(device)
		ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))

	text_aug_path = "/data/home/wangyouze/dataset/flickr30k/Retrieval_flickr_texts_aug.txt"
	text_aug_list = []
	with open(text_aug_path, 'r', encoding='utf-8') as f_aug:
		for line in f_aug:
			new_line = line.strip().split('\t')
			new_line = new_line[1:]
			text_aug_list.append(new_line)


	token_error_rate_list = []
	sentences_similarity = []
	
	for images, aug_images, _, neg_image, texts, neg_text, texts_ids, image_name in tqdm(test_loader):
		
		texts_ids = np.array(texts_ids)
		images = images.to('cuda:0')
		aug_images = aug_images.to('cuda:0')
		neg_image = neg_image.to('cuda:0')
		aug_texts = text_aug_list[texts_ids[0]:texts_ids[-1]+1]
		
		images_ids = [test_loader.dataset.txt2img[i.item()] for i in texts_ids]
	   
		if args.adv != 0:
			adv_texts, adv_images, token_error_rate, noise = multimodal_attack.run(images, aug_images, neg_image, texts, aug_texts, neg_text, embeddings, ref_embeddings,
																		max_length=30, args=args)
			
			adv_images = images_normalize(adv_images)
			sim = calculate_similarity(texts, adv_texts)
			images = adv_images
			texts = adv_texts
		else:
			
			token_error_rate = 0
			sim = 1.0

		sentences_similarity.append(sim)
		# continue
		token_error_rate_list.append(token_error_rate)

		with torch.no_grad():
			# images = images_normalize(images)
			text_input = model_ad.tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
			text_output = model_ad.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
			text_embed = F.normalize(model_ad.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
			text_embeds.append(text_embed)   
			text_ids.append(text_input.input_ids)
			text_atts.append(text_input.attention_mask)

			image_feat = model_ad.visual_encoder(images)   
			image_embed = model_ad.vision_proj(image_feat[:,0,:])            
			image_embed = F.normalize(image_embed, dim=-1)  
			 
			image_feats[images_ids] = image_feat.cpu().detach()
			image_embeds[images_ids] = image_embed.cpu().detach()


	text_embeds = torch.cat(text_embeds,dim=0).cpu().detach()
	text_ids = torch.cat(text_ids,dim=0)
	text_atts = torch.cat(text_atts,dim=0)
	text_ids[:,0] = model.tokenizer.enc_token_id
	
	sims_matrix = image_embeds @ text_embeds.t()
	score_matrix_i2t = torch.full((num_image, num_text),-100.0).to(device)
	
	num_tasks = utils.get_world_size()
	rank = utils.get_rank() 
	step = sims_matrix.size(0)//num_tasks + 1
	start = rank*step
	end = min(sims_matrix.size(0),start+step)

	for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 5000, header)): 
		topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

		encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
		encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long, device=device)
		with torch.no_grad():
			output = model_ad.text_encoder(text_ids[topk_idx], 
										attention_mask = text_atts[topk_idx],
										encoder_hidden_states = encoder_output,
										encoder_attention_mask = encoder_att,                             
										return_dict = True,
									)
			score = model_ad.itm_head(output.last_hidden_state[:,0,:])[:,1]
			score_matrix_i2t[start+i,topk_idx] = score + topk_sim.to(device)
		
		
	sims_matrix = sims_matrix.t()
	score_matrix_t2i = torch.full((num_text, num_image),-100.0).to(device)
	
	step = sims_matrix.size(0)//num_tasks + 1
	start = rank*step
	end = min(sims_matrix.size(0),start+step)    
	
	for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 5000, header)): 
		
		topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
		encoder_output = image_feats[topk_idx.cpu()].to(device)
		encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
		with torch.no_grad():
			output = model_ad.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
										attention_mask = text_atts[start+i].repeat(config['k_test'],1),
										encoder_hidden_states = encoder_output,
										encoder_attention_mask = encoder_att,                             
										return_dict = True,
									)
			score = model_ad.itm_head(output.last_hidden_state[:,0,:])[:,1]
			score_matrix_t2i[start+i,topk_idx] = score + topk_sim.to(device)
		
	result = itm_eval(score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), txt2img=test_dataset.txt2img, img2txt=test_dataset.img2txt)

	print(result)
	print('token_error_rate:', sum(token_error_rate_list)/len(token_error_rate_list))
	print('similarity:', sum(sentences_similarity)/len(sentences_similarity))
  



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="White-box attack.")
	
	parser.add_argument("--start_index", default=0, type=int,
						help="starting sample index")
	parser.add_argument("--num_iters", default=10, type=int,
						help="number of epochs to train for")
	parser.add_argument("--attack_target", default="premise", type=str,
						choices=["premise", "hypothesis"],
						help="attack either the premise or hypothesis for MNLI")
	parser.add_argument("--initial_coeff", default=13, type=int,
						help="initial log coefficients")
	parser.add_argument("--adv_loss", default="ce", type=str,
						choices=["cw", "ce"],
						help="adversarial loss")
	parser.add_argument("--constraint", default="bertscore_idf", type=str,
						choices=["cosine", "bertscore", "bertscore_idf"],
						help="constraint function")
	parser.add_argument("--lr", default=3e-1, type=float,
						help="learning rate")#3e-1 ##1e-1
	parser.add_argument("--kappa", default=5, type=float,
						help="CW loss margin")
	parser.add_argument("--embed_layer", default=-1, type=int,
						help="which layer of LM to extract embeddings from")
	parser.add_argument("--lam_sim", default=1, type=float,
						help="embedding similarity regularizer")
	parser.add_argument("--lam_perp", default=1, type=float,
						help="(log) perplexity regularizer")
	parser.add_argument("--print_every", default=1, type=int,
						help="print loss every x iterations")
	parser.add_argument("--gumbel_samples", default=100, type=int,
						help="number of gumbel samples; if 0, use argmax")

	parser.add_argument("--test_file",
						default="/data/home/wangyouze/dataset/flickr30k/flickr30k_test.json",
						type=str)
	parser.add_argument("--image_root", default="/data/home/wangyouze/dataset/flickr30k/", type=str)
	parser.add_argument("--batch_size_test", default=150, type=int)#25

	parser.add_argument('--config', default='/data/home/wangyouze/projects/VLP_attack/configs/Retrieval_flickr.yaml')
	parser.add_argument('--output_dir', default='output/retrieval')
	parser.add_argument('--text_encoder', default="/data/home/wangyouze/MultimodalAttack/bert-base-uncased/")
	parser.add_argument('--image_encoder', default='ViT-B/16')
	parser.add_argument('--seed', default=2023, type=int)
	parser.add_argument('--alpha', default=1.6, type=float)
	parser.add_argument('--eps', default=2.0, type=float)
	parser.add_argument('--adv', default=4, type=int)

	args = parser.parse_args()
	print_args(args)

	config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
	main(args, config)