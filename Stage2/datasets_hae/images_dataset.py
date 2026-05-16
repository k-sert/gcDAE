from torch.utils.data import Dataset
from PIL import Image
import random
from utils import data_utils
from configs import data_configs
import os
import sys
import cv2
sys.path.append(".")
from configs import data_configs
import numpy as np


class ImagesDataset(Dataset):

	def __init__(self, dataset_type, split='train'):
		if dataset_type not in data_configs.DATASETS.keys():
			raise Exception(f'{dataset_type} is not a valid dataset_type')
		if split not in ['train', 'val', 'test']:
			raise Exception(f'{split} is not a valid split')
		print(f'Loading {split} dataset for {dataset_type}')
		dataset_args = data_configs.DATASETS[dataset_type]
		transforms_dict = dataset_args['transforms']().get_transforms()
		source_root, target_root = self._get_split_roots(dataset_args, split)
		if split == 'train':
			target_transform = transforms_dict['transform_gt_train']
			source_transform = transforms_dict['transform_source']
		else:
			target_transform = transforms_dict.get('transform_gt_test', transforms_dict['transform_test'])
			source_transform = transforms_dict['transform_test']
		train_transform = transforms_dict['transform_train']
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.classes, self.class_to_idx = data_utils.find_classes(source_root)
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.train_transform = train_transform

	def _get_split_roots(self, dataset_args, split):
		if split == 'train':
			return dataset_args['train_source_root'], dataset_args['train_target_root']
		source_key = f'{split}_source_root'
		target_key = f'{split}_target_root'
		if source_key not in dataset_args and split == 'val':
			source_key = 'test_source_root'
			target_key = 'test_target_root'
		return dataset_args[source_key], dataset_args[target_key]

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		# item = self.get_sample(index)
		# return item
		while (True):
			try:
				item = self.get_sample(index)
				return item
			except:
				index = np.random.randint(0, len(self.source_paths)-1)

	def get_sample(self, index):
		from_path = self.source_paths[index]
		class_label = from_path.split('/')[-2]
		label = self.class_to_idx[class_label]
		from_im = cv2.imread(from_path)
		from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
		# from_im = Image.open(from_path)
		# from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = cv2.imread(to_path)
		to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)
		# to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(image=to_im)["image"]
			to_im = np.array(to_im).astype(np.uint8)
			to_im = (to_im / 127.5 - 1.0).astype(np.float32)
			# to_im = Image.fromarray(to_im["image"])

		if self.source_transform:
			from_im = self.source_transform(image=from_im)["image"]
			from_im = np.array(from_im).astype(np.uint8)
			from_im = (from_im / 127.5 - 1.0).astype(np.float32)
			# from_im = Image.fromarray(from_im["image"])
		else:
			from_im = to_im
		from_im = np.transpose(from_im, (2, 0, 1))
		to_im = np.transpose(to_im, (2, 0, 1))

		return from_im, to_im, label
