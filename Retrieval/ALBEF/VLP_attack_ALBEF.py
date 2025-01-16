import torch
from transformers import BatchEncoding
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from jiwer import wer
import torch.nn as nn
from scipy import spatial
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet
import random
from enum import Enum




filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)
def equal_normalize(x):
    return x

class NormType(Enum):
    Linf = 0
    L2 = 1

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta


class input_text():
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class MultiModalAttacker(nn.Module):
    def __init__(self, model, visual_encoder_1, visual_encoder_2, ref_net, idf_dict, tokenizer, end_idx=5000,
                 img_decay=1.0, text_decay=1.0, transformation_num=5, device=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_net
        self.repeat = 1
        self.image_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))
        self.idf_dict = idf_dict
        self.end_idx = end_idx
        self.img_decay = img_decay
        self.text_decay = text_decay
        self.temp = 0.07  #0.07
        self.inference_image_1 = visual_encoder_1
        self.inference_image_2 = visual_encoder_2

        self.device = device
        self.m = transformation_num
        self.resize_rate = 0.9
        self.diversity_prob = 0.5
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)


    def log_perplexity(self, logits, coeffs):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_coeffs = coeffs[:, 1:, :].contiguous()
        shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
        return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()

    def bert_score(self, refs, cands, weights=None):
        refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
        if weights is not None:
            refs_norm *= weights[:, None]
        else:
            refs_norm /= refs.size(1)
        cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
        cosines = refs_norm @ cands_norm.transpose(1, 2)
        # remove first and last tokens; only works when refs and cands all have equal length (!!!)
        cosines = cosines[:, 1:-1, 1:-1]
        R = cosines.max(-1)[0].sum(1)
        return R

    def info_nce_loss(self, features_1, features_2):

        labels = torch.cat([torch.arange(features_1.size(0)) for i in range(1)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda:0')

        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)

        similarity_matrix = torch.matmul(features_1, features_2)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda:0')
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda:0')

        logits = logits / 1
        return logits, labels

   
    def info_nce_loss_2(self, features, neg_features, batch_size, n_views):

        labels = torch.cat([torch.arange(1) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda')

        features = F.normalize(features, dim=1)
        neg_features = F.normalize(neg_features, dim=0)
        similarity_matrix = torch.matmul(features, features.permute(0, 2, 1))
        negatives = torch.matmul(features, neg_features.permute(0, 2, 1))
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda:0')
        labels = labels[~mask].view(labels.shape[0], -1)

        mask = mask.repeat(batch_size, 1, 1)
        labels = labels.repeat(batch_size, 1, 1)
        # positives_1 = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], similarity_matrix.shape[1], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], labels.shape[1], -1)

        # select only the negatives the negatives
        # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([negatives, positives], dim=2)
        labels = torch.zeros((logits.shape[0], logits.shape[1]), dtype=torch.long).to('cuda:0')
        logits = logits / 0.25
        return logits, labels
    def contrastive_loss(self, anchor, positive, negative, temperature=0.1):
        pos_dot_producr = torch.sum(anchor * positive, dim=1, keepdim=True)
        neg_dot_product = torch.matmul(anchor, negative.T)
        logits = torch.cat([pos_dot_producr, neg_dot_product], dim=1)
        logits /= temperature

        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, torch.zeros_like(log_probs[:, 0], dtype=torch.long))
        return loss

    def regularize(self, x):
        reg = 1e-6
        return (0.5 * reg * torch.sum(torch.pow(x, 2)))
    def gen_text_augument(self, log_coeffs, embeddings):
        coeffs_aug_3 = F.gumbel_softmax(log_coeffs, hard=False)
        inputs_text_embeds_aug_3 = (coeffs_aug_3 @ embeddings[None, :, :])
        text_features_aug_3 = F.normalize(inputs_text_embeds_aug_3, dim=-1)
        words_features_aug_3 = text_features_aug_3 / text_features_aug_3.norm(dim=-1, keepdim=True)
        text_features_aug_3 = words_features_aug_3.mean(1)
        return text_features_aug_3
    def gen_text_augment_2(self, text, n):
        new_words = text.copy()
        random_word_list = list(set([word for word in text if word not in filter_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            # get_synonyms 获取某个单词的同义词列表

            synonyms = []

            for syn in wordnet.synsets(random_word):
                for lm in syn.lemmas():
                    synonyms.append(lm.name())

            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        sentence = ' '.join(new_words)
        return sentence

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x
    def run(self, image, aug_images, neg_image, text, aug_texts, neg_text, embeddings, ref_embeddings, max_length=30, args={}):
        batch_size = image.size(0)
        end_index = self.end_idx
        adv_losses, ref_losses, perp_losses, entropies = torch.zeros(end_index - args.start_index,
                                                                     args.num_iters), torch.zeros(
            end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index,
                                                                       args.num_iters), torch.zeros(
            end_index - args.start_index, args.num_iters)

        device = self.device
        sentence_length = []
        for sentence in text:
            sentence = sentence.split(' ')
            sentence_length.append(len(sentence))
        text_input_ids_ = self.tokenizer(text, padding='max_length', max_length=max_length, truncation=True,
                                         return_tensors="pt")
        text_input_ids = text_input_ids_.input_ids.to('cuda:0')

        aug_text_features = []
      

        with torch.no_grad():
            for k, aug_sentences in enumerate(aug_texts):
                aug_sentences_tokens = self.tokenizer(aug_sentences, padding='max_length', truncation=True, max_length=30,
                                return_tensors="pt").to(device)
                aug_text_features.append(self.model.inference(image[k,:].repeat(len(aug_sentences), 1, 1, 1).to(device), aug_sentences_tokens, use_embeds=False)['text_feat'])
            text_features_aug = torch.stack(aug_text_features, dim=0)



        with torch.no_grad():
            neg_text_tokens = self.tokenizer(neg_text, padding='max_length', truncation=True, max_length=30,
                                return_tensors="pt").to(device)
            neg_text_logit = self.model.inference(neg_image.to(device), neg_text_tokens, use_embeds=False)
            neg_text_features = neg_text_logit['text_feat'].unsqueeze(1)
        with torch.no_grad():
            text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=30,
                                return_tensors="pt").to(device)
            clean_logit = self.model.inference(image.to(device), text_tokens, use_embeds=False)
            origin_text_features = clean_logit['text_feat']
            origin_image_features = clean_logit['image_feat']


        with torch.no_grad():
            orig_output = self.ref_model(text_input_ids.to('cuda:0')).logits
            # .hidden_states[args.embed_layer]
            if args.constraint.startswith('bertscore'):
                if args.constraint == "bertscore_idf":
                    text_input_ids_tmp = text_input_ids.cpu().detach().numpy()
                    ref_weights = []
                    for line in text_input_ids_tmp:
                        tmp = []
                        for idx in line:
                            tmp.append(self.idf_dict[idx])
                        ref_weights.append(tmp)
                    ref_weights = torch.FloatTensor(ref_weights).to(device).type(
                        torch.float32)
                    ref_weights = F.normalize(ref_weights, p=2, dim=1)

            elif args.constraint == 'cosine':
                # GPT-2 reference model uses last token embedding instead of pooling
                if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                    orig_output = orig_output[:, 0, :]

        log_coeffs = torch.zeros(text_input_ids.size(0), text_input_ids.size(1), embeddings.size(0)).to(device)
        # log_coeffs.scatter(1, text_input_ids, args.initial_coeff)

        indices = torch.arange(text_input_ids.size(1)).long()
        for i in range(text_input_ids.size(0)):
            log_coeffs[i, indices, text_input_ids[i, :]] = args.initial_coeff

        optimizer = torch.optim.RMSprop([log_coeffs], lr=args.lr, weight_decay=1e-11, momentum=0.6)
        # optimizer = torch.optim.Adam([log_coeffs], lr=args.lr, weight_decay=5e-12)

        forbidden = np.zeros((text_input_ids.size(0), text_input_ids.size(1))).astype('bool')
        # set [CLS] and [SEP] tokens to forbidden
        forbidden[:, 0] = True
        forbidden[:, -1] = True
        offset = 0
        forbidden[(max_length - offset):] = True
        forbidden_indices = np.repeat(np.arange(0, len(text_input_ids)), text_input_ids.size(0))
        forbidden_indices = torch.from_numpy(forbidden_indices).to(device)

        ori_image = image.data
        eps = args.eps / 255.
        alpha = args.alpha / 255.
       
        image_momentum = torch.zeros_like(image).detach().to('cuda:0')
        text_momentum = torch.zeros_like(log_coeffs).detach().to(device)
        cross_entropy_loss = nn.CrossEntropyLoss()

        image_features_aug = []
        with torch.no_grad():
            for b in range(image.size(0)):
                image_features_aug.append(self.inference_image_2(aug_images[b, :])['image_feat'].unsqueeze(0).to('cuda:0'))
        image_features_aug = torch.cat(image_features_aug, dim=0).type(self.model.visual_encoder.pos_embed.dtype)
        neg_image_features = self.inference_image_2(neg_image)['image_feat'].unsqueeze(1).to('cuda:0').type(self.model.visual_encoder.pos_embed.dtype)

        text_features_all = torch.cat([ origin_text_features.unsqueeze(1),
                                        text_features_aug],
                                    dim=1).type(self.model.visual_encoder.pos_embed.dtype)

        image_features_all = torch.cat(
            [origin_image_features.unsqueeze(1), image_features_aug], dim=1)

        for i in range(args.num_iters):
            image.to('cuda:0')
            image.requires_grad = True
            log_coeffs.requires_grad = True
            image_grad = torch.zeros_like(image).detach()
            # log_coeffs_grad = torch.zeros_like(log_coeffs).detach()
            neg_image_features = neg_image_features.detach()
            neg_text_features = neg_text_features.detach()
            for _ in torch.arange(self.m):
                coeffs = F.gumbel_softmax(log_coeffs, hard=False)  # B x T x V
                inputs_text_embeds = (coeffs @ embeddings[None, :, :])  # B x T x D

                # text_features = F.normalize(inputs_text_embeds, dim=-1)
                image_features = self.inference_image_1(self.image_normalize(self.input_diversity(image)))['image_feat'].to('cuda:0')
                # image_features = image_features_[:, 0, :]

                adv_image_features = (image_features / image_features.norm(dim=-1, keepdim=True)).unsqueeze(1)
                adv_words_features = inputs_text_embeds / inputs_text_embeds.norm(dim=-1, keepdim=True)

                adv_text_features = F.normalize(self.model.text_proj(adv_words_features[:, 0, :]), dim=-1).type(self.model.visual_encoder.pos_embed.dtype).unsqueeze(1)
                # adv_text_features = adv_words_features.mean(1).unsqueeze(1).type(self.model.visual_encoder.pos_embed.dtype)

                i2i_logits, i2i_labels = self.info_nce_loss_2(
                    torch.cat([adv_image_features, image_features_all], dim=1), neg_features=neg_image_features, batch_size=batch_size, n_views=image_features_all.size(1)+1)
                loss_ita = cross_entropy_loss(i2i_logits, i2i_labels)

                adv_text_inputs = coeffs.argmax(2)
                adv_text_inputs = adv_text_inputs[offset:len(adv_text_inputs) - offset].cpu().tolist()
                adv_text_inputs_ = torch.LongTensor(adv_text_inputs).to('cuda:0')

                adv_loss = -self.cosine_similarity(adv_image_features.squeeze(1), adv_text_features.squeeze(1)).mean()
                


                # Similarity constraint
                adv_bert_features = self.ref_model(adv_text_inputs_).logits
                ref_loss = -args.lam_sim * self.bert_score(orig_output, adv_bert_features).mean()
                
                ref_embeds = (coeffs @ ref_embeddings[None, :, :])
                pred = self.ref_model(inputs_embeds=ref_embeds)
                perp_loss = args.lam_perp * self.log_perplexity(pred.logits, coeffs)

                ti_logits, ti_labels = self.info_nce_loss_2(torch.cat([ adv_image_features,text_features_all], dim=1), neg_features=neg_text_features,
                                                            batch_size=batch_size, n_views=text_features_all.size(1) + 1)
                adv_it_loss_1 = cross_entropy_loss(ti_logits, ti_labels)

                it_logits, it_labels = self.info_nce_loss_2(torch.cat([adv_text_features, image_features_all], dim=1),
                                                            neg_features=neg_image_features,
                                                            batch_size=batch_size, n_views=image_features_all.size(1) + 1)
                adv_it_loss_2 = cross_entropy_loss(it_logits, it_labels)
                adv_it_loss = adv_it_loss_1 + adv_it_loss_2

                total_loss = 8*adv_loss \
                            + ref_loss \
                            + perp_loss \
                             - 10 * torch.norm(adv_image_features, p=2) \
                             + adv_it_loss\
                             + loss_ita \
                            
                    # +8*adv_it_loss
                total_loss /= batch_size
                total_loss.backward()
                image_grad += image.grad

                # log_coeffs_grad += log_coeffs.grad


                log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
                optimizer.step()

            image_grad = image_grad / self.m

            # image_grad = image_grad.grad
            image_grad = image_grad / torch.mean(torch.abs(image_grad), dim=(1, 2, 3), keepdim=True)
            image_grad = image_grad * self.img_decay + image_momentum
            image_momentum = image_grad
            image = image.detach() + alpha * image_grad.sign()
            #
            delta = torch.clamp(image - ori_image, min=-eps, max=eps)
            image = torch.clamp(ori_image + delta, min=0, max=1).detach()

            # Log statistics
            adv_losses[0, i] = adv_loss.detach().item()
            ref_losses[0, i] = ref_loss.detach().item()
            perp_losses[0, i] = perp_loss.detach().item()
            # entropies[0, i] = entropy.detach().item()

        # print('clean_text:', text)
        token_errors = []
        adv_texts = []
        adv_logits = []
        adv_log_coeffs = []
        # print('ADVERSARIAL TEXT')

        adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(2)

        adv_ids = adv_ids[offset:len(adv_ids) - offset].cpu().tolist()

        batch_size = len(adv_ids)
        adv_texts = []
        for t in range(batch_size):
            adv_id = adv_ids[t]
            adv_text = self.tokenizer.decode(adv_id).replace('[CLS]', '').split('[PAD]')[0][1:-1]
            # x = self.tokenizer(adv_text, max_length=max_length, truncation=True, return_tensors='pt')
            if adv_text == '':
                adv_text = text[t]
            token_errors.append(wer(adv_text, text[t]))
            adv_texts.append(adv_text)
        print(adv_texts)
        # adv_output = self.model.inference(image, adv_texts)

        # remove special tokens from adv_log_coeffs
        adv_log_coeffs.append(log_coeffs[offset:(log_coeffs.size(0) - offset), :].cpu())  # size T x V

        return adv_texts, image, sum(token_errors) / len(token_errors), delta

