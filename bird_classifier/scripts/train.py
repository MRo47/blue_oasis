import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from bird_classifier.augment import spec_augment, time_domain_augment
from bird_classifier.data_splitter import filter_classes, stratified_group_split
from bird_classifier.dataset import BirdSoundDataset
from bird_classifier.logging_config import logger
from bird_classifier.preprocessing import preprocess
from sklearn.metrics import classification_report, f1_score
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_class_weights(df, label_col, class_to_idx):
    counts = df[label_col].value_counts()
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)  # normalize
    weight_tensor = torch.tensor([weights[c] for c in class_to_idx.keys()], dtype=torch.float32)
    return weight_tensor

def validate(val_loader, model, criterion, device):
	model.eval()
	all_preds = []
	all_labels = []
	running_loss = 0

	with torch.no_grad():
		for images, labels in val_loader:
			images, labels = images.to(device), labels.to(device)
			output = model(images)
			preds = torch.argmax(output, dim=1)
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())
			loss = criterion(output, labels)
			running_loss += loss.item()

	loss = running_loss / len(val_loader)
	macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
	return loss, macro_f1

def log_epoch_metrics(epoch, train_loss, val_loss, f1, class_weights):
    logger.info(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, macro_f1={f1:.4f}")
    logger.info(f"Max class weight: {class_weights.max():.2f}, Min class weight: {class_weights.min():.2f}")

def evaluate_full(model, val_loader, device, classes):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=False)
    logger.info("\n" + report)

def parse_args():
	parser = argparse.ArgumentParser(
		description="Train bird species classifier on bioacoustic data"
	)

	# Paths
	parser.add_argument(
		"--annotations_csv",
		type=Path,
		help="Path to the annotations CSV file",
	)
	parser.add_argument(
		"--recordings_path",
		type=Path,
		help="Path to the directory containing audio recordings",
	)
	parser.add_argument(
		"--save_path",
		type=Path,
		help="Directory to save model checkpoints and logs",
	)

	# Audio / training parameters
	parser.add_argument("--sample_rate", type=int, default=22050, help="Target sample rate (Hz)")
	parser.add_argument("--call_duration", type=float, default=3.0, help="Clip duration in seconds")
	parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
	parser.add_argument("--min_samples", type=int, default=10, help="Minimum number of samples per class")
	parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")

	# Optional extras
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
	parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Computation device")

	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	annotations_csv = args.annotations_csv
	recordings_path = args.recordings_path
	sample_rate = args.sample_rate
	call_duration = args.call_duration
	batch_size = args.batch_size
	min_samples = args.min_samples
	num_epochs = args.num_epochs
	save_path = args.save_path
	learning_rate = args.lr
	num_workers = args.num_workers
	device = torch.device(args.device)

	logger.info(f"Annotations CSV: {annotations_csv}")
	logger.info(f"Recordings Path: {recordings_path}")
	logger.info(f"Sample Rate: {sample_rate}")
	logger.info(f"Call Duration: {call_duration}")
	logger.info(f"Batch Size: {batch_size}")
	logger.info(f"Min Samples: {min_samples}")
	logger.info(f"Num Epochs: {num_epochs}")
	logger.info(f"Save Path: {save_path}")
	logger.info(f"Learning Rate: {learning_rate}")
	logger.info(f"Num Workers: {num_workers}")
	logger.info(f"Device: {device}")

	if not save_path.exists():
		logger.info(f"Creating save path: {save_path}")
		save_path.mkdir(parents=True)

	df = pd.read_csv(annotations_csv)

	# filter out classes with less than min_samples
	df = filter_classes(df,  min_sample_size=min_samples)

	train_df, val_df, _ = stratified_group_split(
		df,
		label_id='Species eBird Code',
		group_id='Filename',
		test_size=0.2,
		val_size=0.2,
	)

	train_dataset = BirdSoundDataset(
		df=train_df,
		recordings_path=recordings_path,
		mode='train',
		sr=sample_rate,
		duration=call_duration,
		preprocess_fn=preprocess,
		time_augment_fn=time_domain_augment,
		spec_augment_fn=spec_augment
	)

	# validation dataset with NO augmentations
	val_dataset = BirdSoundDataset(
		df=val_df,
		recordings_path=recordings_path,
		mode='valid', 
		sr=sample_rate,
		duration=call_duration,
		preprocess_fn=preprocess,
		time_augment_fn=None,
		spec_augment_fn=None
	)

	# data loaders
	train_loader = DataLoader(
		train_dataset, 
		batch_size=batch_size, 
		shuffle=True, 
		num_workers=num_workers,
		pin_memory=True
	)

	val_loader = DataLoader(
		val_dataset, 
		batch_size=batch_size * 2,
		shuffle=False, 
		num_workers=num_workers,
		pin_memory=True
	)

	# report dataset info
	logger.info(f"Number of training samples: {len(train_dataset)}")
	logger.info(f"Number of validation samples: {len(val_dataset)}")
	logger.info(f"Number of classes: {len(train_dataset.classes)}")

	# model
	model = create_model(
		"convnext_tiny", 
		pretrained=True, 
		in_chans=1, 
		num_classes=len(train_dataset.classes)
	).to(device)

	# calculate class weights
	class_weights = compute_class_weights(train_df, 'Species eBird Code', train_dataset.class_to_idx)
	logger.info(f"Class weights summary: max={class_weights.max():.2f}, min={class_weights.min():.2f}")
	
	# weighted cross-entropy loss
	criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

	# training loop
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0

		for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		scheduler.step()
		val_loss, macro_f1 = validate(val_loader, model, criterion, device)
		log_epoch_metrics(epoch, total_loss/len(train_loader), val_loss, macro_f1, class_weights)

		if macro_f1 > best_f1:
			best_f1 = macro_f1
			torch.save(model.state_dict(), save_path / "best_model.pt")
			logger.info(f"saved new best model (f1={best_f1:.4f})")

	logger.info("Training complete!")
	evaluate_full(model, val_loader, device, train_dataset.classes)

if __name__ == "__main__":
	main()