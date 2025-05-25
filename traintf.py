import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

def build_model(num_classes, input_shape=(224, 224, 3), lr=0.0005, fine_tune=False):
    """Build MobileNetV2 model with transfer learning and optional fine-tuning."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    
    if fine_tune:
        for layer in base_model.layers[-20:]:
            layer.trainable = True
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a CNN model for brain cancer classification using TensorFlow.")
    parser.add_argument("--data_dir", type=str, default="brain_cancer/training",
                        help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, default="Zainab",
                        help="Name for saving the model")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes in the dataset (e.g., meningioma, glioma, pituitary, notumor)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Initial learning rate for the optimizer")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory to save the model and plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--fine_tune_epoch", type=int, default=20,
                        help="Epoch to start fine-tuning (0 to disable fine-tuning)")
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_data_generators(data_dir, batch_size, validation_split, input_size=224):
    """Create data generators for training and validation with enhanced augmentation."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        zoom_range=[0.8, 1.2],
        shear_range=0.2,
        validation_split=validation_split,
        preprocessing_function=lambda x: np.stack([x[:,:,0]]*3, axis=-1) if len(x.shape) == 3 and x.shape[-1] == 1 else x
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    train_labels = train_generator.classes
    val_labels = validation_generator.classes
    print("Training class distribution:", Counter(train_labels))
    print("Validation class distribution:", Counter(val_labels))
    print(f"Found {train_generator.samples} training images.")
    print(f"Found {validation_generator.samples} validation images.")
    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("No images found in training or validation set. Check your data directory.")

    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in class_counts.items()}
    print("Class weights:", class_weights)

    return train_generator, validation_generator, class_weights

def train_model(model, train_generator, validation_generator, num_epochs, class_weights=None):
    """Train the model with early stopping, learning rate scheduling, and model checkpointing."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'Zainab_best_model.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=num_epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights,
        verbose=1
    )
    return history

def save_model(model, save_path):
    """Save the trained model in TensorFlow Keras format with fallback to HDF5."""
    os.makedirs(save_path, exist_ok=True)
    try:
        model.save(save_path, save_format='tf')
        print(f"TensorFlow SavedModel saved in directory: {save_path}")
    except Exception as e:
        print(f"Failed to save as SavedModel: {e}")
        h5_path = os.path.join(save_path, "Zainab_model.h5")
        model.save(h5_path)
        print(f"Model saved in HDF5 format: {h5_path}")

def plot_metrics(history, save_dir, model_name, validation_generator, num_classes, model):
    """Plot training and validation metrics and confusion matrix."""
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name}_metrics_plot.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Metrics plot saved as {plot_path}")
    plt.close()

    validation_generator.reset()
    y_pred = model.predict(validation_generator, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(),
                yticklabels=validation_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved as {cm_path}")
    plt.close()

def main():
    print("Current working directory:", os.getcwd())
    args = parse_arguments()
    set_seed(args.seed)
    train_generator, validation_generator, class_weights = create_data_generators(
        args.data_dir,
        args.batch_size,
        args.validation_split
    )
    
    model = build_model(
        args.num_classes,
        lr=args.lr,
        fine_tune=False
    )
    
    history = train_model(
        model,
        train_generator,
        validation_generator,
        args.fine_tune_epoch if args.fine_tune_epoch > 0 else args.num_epochs,
        class_weights
    )
    
    if args.fine_tune_epoch > 0 and args.fine_tune_epoch < args.num_epochs:
        print("Starting fine-tuning...")
        model = build_model(
            args.num_classes,
            lr=args.lr / 10,
            fine_tune=True
        )
        model.load_weights('Zainab_best_model.weights.h5')
        history_fine = train_model(
            model,
            train_generator,
            validation_generator,
            args.num_epochs - args.fine_tune_epoch,
            class_weights
        )
        for metric in ['loss', 'accuracy', 'precision', 'recall']:
            history.history[metric].extend(history_fine.history[metric])
            history.history[f'val_{metric}'].extend(history_fine.history[f'val_{metric}'])
    
    save_path = os.path.join(args.save_dir, f"{args.model_name}_model.tensorflow")
    save_model(model, save_path)
    plot_metrics(history, args.save_dir, args.model_name, validation_generator, args.num_classes, model)
    final_metrics = {metric: history.history[f'val_{metric}'][-1] for metric in ['loss', 'accuracy', 'precision', 'recall']}
    for metric, value in final_metrics.items():
        print(f"Final validation {metric}: {value:.4f}")

if __name__ == "__main__":
    main()