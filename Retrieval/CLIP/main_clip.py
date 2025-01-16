import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import matplotlib.pyplot as plt
from bert_score.utils import get_idf_dict
import transformers
import argparse
import numpy as np
sys.path.append('/data/home/wangyouze/projects/github/VLP_attack/')
from models.clip import clip
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
from VLP_attack_CLIP import MultiModalAttacker
from tqdm import tqdm
import cv2

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from jiwer import wer
# Load the model
sim_model = SentenceTransformer("/data/home/wangyouze/projects/others/multimodal_machine_translation/declutr-small/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")
        
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

def main(args):

    tokenizer = BertTokenizer.from_pretrained("/data/home/wangyouze/MultimodalAttack/bert-base-uncased/")
    tokenizer.model_max_length = 30

    model, prepocess = clip.load(args.image_encoder, device=device)
    model.set_tokenizer(tokenizer)

    model_ad, _ = clip.load(args.image_encoder, device=device)
    model_ad.set_tokenizer(tokenizer)

    model = model.to(device)
    model_ad = model_ad.to(device)

    visual_encoder_1 = model.encode_image
    visual_encoder_2 = model.encode_image
  
    model.eval()
    model_ad.eval()

    print("Creating dataset")
    n_px = model.visual.input_resolution
    test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])

    test_dataset = pair_dataset(args.test_file, test_transform, args.image_root)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, num_workers=4) # 这个sampler会自动分配数据到各个gpu上
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder).to(device)
    # Compute idf dictionary for BERTScore
    idf_dict = get_idf_dict(test_dataset.text, tokenizer, nthreads=20)
    multimodal_attack = MultiModalAttacker(model, visual_encoder_1, visual_encoder_2, ref_model, idf_dict, tokenizer, end_idx=5000, transformation_num=7, device=device)
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    num_text = len(test_loader.dataset.text)
    print('num_text:', num_text)
    num_image = len(test_loader.dataset.ann)
    print('num_image:', num_image)
    image_feats = torch.zeros(num_image, model.visual.output_dim)
    text_feats = torch.zeros(num_text, model.visual.output_dim)

    with torch.no_grad():
        tokens_id = torch.arange(0, tokenizer.vocab_size).long().cuda()
        embeddings = model.token_embedding(tokens_id).to(device)
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
    run_time = []

    f_res = open("/data/home/wangyouze/projects/save/VLP_attack/Retrieval/flickr30k/VLP_attack_B_16_iter10_test/results_CLIP.txt", 'a+', encoding='utf-8')

    for images, aug_images, _, neg_image, texts, neg_text, texts_id, image_name in tqdm(test_loader):
        
        texts_id = np.array(texts_id)
        images = images.to('cuda:0')
        aug_images = aug_images.to('cuda:0')
        neg_image = neg_image.to('cuda:0')
        aug_texts = text_aug_list[texts_id[0]:texts_id[-1]+1]
       
        start = time.time()
       
        if args.adv != 0:
            adv_texts, adv_images, token_error_rate, noise = multimodal_attack.run(images, aug_images, neg_image, texts, aug_texts, neg_text, embeddings, ref_embeddings,
                                                                        max_length=30, args=args)
            
            sim = calculate_similarity(texts, adv_texts)
        else:
            adv_images = images
            adv_texts = texts
            token_error_rate = 0
            sim = 1.0

        end = time.time()
        cost_time = end - start
        run_time.append(cost_time)

        sentences_similarity.append(sim)
        # continue
        token_error_rate_list.append(token_error_rate)

        for t in range(adv_images.size(0)):
            # images = images.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            # plt.imshow(images)
            # plt.show()

            fake_img = torch.clamp(adv_images[t,:], 0, 1).permute([1, 2, 0]).data.cpu().numpy()[:, :, [2, 1, 0]]

            # noise = torch.clamp(noise[t, :], 0, 1).permute([1, 2, 0]).data.cpu().numpy()[:, :, [2, 1, 0]]
            # noise = noise * 5
            # plt.imshow(noise)
            # plt.show()
            cv2.imwrite('/data/home/wangyouze/projects/save/VLP_attack/Retrieval/flickr30k/VLP_attack_B_16_iter10_test/fake_images/fake_{}.png'.format(texts_id[t]), (fake_img * 255).astype(int))



        images_ids = [test_loader.dataset.txt2img[i.item()] for i in texts_id]
        with torch.no_grad():
            adv_images = images_normalize(adv_images).to('cuda:0')
            output = model_ad.inference(adv_images, adv_texts)

            image_feats[images_ids] = output['image_feat'].cpu().float().detach()
            text_feats[texts_id] = output['text_feat'].cpu().float().detach()
             

        for j, text_id in enumerate(texts_id):
            f_res.write(str(texts_id[j]) + '\t' + texts[j] + '\t' + adv_texts[j] + '\n')

           
    print('run_time:', sum(run_time)/len(run_time))


    sims_matrix = image_feats @ text_feats.t()
    
    result = itm_eval(sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy(), test_dataset.img2txt, test_dataset.txt2img)
    print(result)
    print('token_error_rate:', sum(token_error_rate_list)/len(token_error_rate_list))
    print('similarity:', sum(sentences_similarity)/len(sentences_similarity))
    # with open('./results/Retrieval_CLIP_attack_results.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(result))
    # f.close()



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
    parser.add_argument("--batch_size_test", default=50, type=int)#25

    parser.add_argument('--config', default='/data/home/wangyouze/projects/VLP_attack/configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/retrieval')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--image_encoder', default='ViT-B/16')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--alpha', default=1.6, type=float)
    parser.add_argument('--eps', default=16.0, type=float)
    parser.add_argument('--adv', default=1, type=int)

    args = parser.parse_args()
    print_args(args)
    main(args)