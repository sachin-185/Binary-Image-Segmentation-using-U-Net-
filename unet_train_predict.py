import os
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
#  LOSS FUNCTIONS
# -------------------------------------------------------------
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

# -------------------------------------------------------------
#  DATALOADER
# -------------------------------------------------------------
class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, img_list, mask_list, img_size=256, batch_size=4, is_training=False):
        self.img_list = img_list
        self.mask_list = mask_list
        self.img_size = img_size
        self.batch_size = batch_size
        self.is_training = is_training

    def __len__(self):
        return len(self.img_list) // self.batch_size

    def apply_augmentation(self, img, mask):
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

        if tf.random.uniform(()) > 0.5:
            img_tensor = tf.image.flip_left_right(img_tensor)
            mask_tensor = tf.image.flip_left_right(mask_tensor)

        img_tensor = tf.image.random_brightness(img_tensor, 0.2)
        img_tensor = tf.image.random_saturation(img_tensor, 0.8, 1.2)

        img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)
        mask_tensor = tf.round(mask_tensor)

        return img_tensor.numpy(), mask_tensor.numpy()

    def __getitem__(self, index):
        imgs, masks = [], []
        batch_imgs = self.img_list[index * self.batch_size:(index + 1) * self.batch_size]
        batch_masks = self.mask_list[index * self.batch_size:(index + 1) * self.batch_size]

        for img_path, mask_path in zip(batch_imgs, batch_masks):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype("float32") / 255.0

            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = np.expand_dims(mask, axis=-1) / 255.0

            imgs.append(img)
            masks.append(mask)

        batch_x, batch_y = np.array(imgs), np.array(masks)

        if self.is_training:
            aug_x, aug_y = [], []
            for img, mask in zip(batch_x, batch_y):
                a, b = self.apply_augmentation(img, mask)
                aug_x.append(a)
                aug_y.append(b)
            return np.array(aug_x), np.array(aug_y)

        return batch_x, batch_y

# -------------------------------------------------------------
#  U-NET MODEL
# -------------------------------------------------------------
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = tf.keras.layers.Input(input_shape)

    c1 = conv_block(inputs, 64)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    c5 = conv_block(p4, 1024)

    u6 = tf.keras.layers.UpSampling2D()(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = tf.keras.layers.UpSampling2D()(c6)
    u7 = tf.keras.layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = tf.keras.layers.UpSampling2D()(c7)
    u8 = tf.keras.layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = tf.keras.layers.UpSampling2D()(c8)
    u9 = tf.keras.layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c9)
    return tf.keras.Model(inputs, outputs)

# -------------------------------------------------------------
#  MAIN FUNCTION
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--img", type=str, help="Path to image for prediction")
    parser.add_argument("--model", type=str, default="unet_vehicle.h5")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # ------------------ PREDICTION MODE ------------------
    # ------------------ PREDICTION MODE ------------------
if args.img:
    print("✅ Loading model...")
    model = tf.keras.models.load_model(args.model, compile=False)

    img = cv2.imread(args.img)
    if img is None:
        print(f"❌ Error: Cannot load image '{args.img}'. Check path or file format.")
        exit()

    orig = img.copy()
    img = cv2.resize(img, (args.img_size, args.img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0, :, :, 0]
    prob_map = (pred * 255).astype(np.uint8)
    mask = (pred > args.threshold).astype(np.uint8) * 255

    cv2.imwrite("pred_prob.png", prob_map)
    cv2.imwrite("pred_mask.png", mask)
    print("✅ Saved: pred_prob.png, pred_mask.png")
    exit()


    # ------------------ TRAINING MODE ------------------
    img_paths = sorted(glob.glob(os.path.join(args.data_dir, "images", "*")))
    mask_paths = sorted(glob.glob(os.path.join(args.data_dir, "masks", "*")))
    assert len(img_paths) == len(mask_paths), "❌ Image/Mask count mismatch!"

    print(f"✅ Found {len(img_paths)} samples")

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_gen = DataLoader(train_imgs, train_masks, args.img_size, args.batch, True)
    val_gen = DataLoader(val_imgs, val_masks, args.img_size, args.batch, False)

    model = build_unet(input_shape=(args.img_size, args.img_size, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=bce_dice_loss, metrics=[dice_coef])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5)
    ]

    model.fit(train_gen, validation_data=val_gen,
              epochs=args.epochs, verbose=2, callbacks=callbacks)

    print("\n✅ Training complete. Best model saved as", args.model)
