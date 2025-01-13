from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# تعطيل التدوين العلمي للأرقام لتسهيل القراءة
np.set_printoptions(suppress=True)

# تحميل النموذج
model_path = "/mnt/data/extracted_model/keras_model.h5"
model = load_model(model_path, compile=False)

# تحميل أسماء الفئات
labels_path = "/mnt/data/extracted_model/labels.txt"
class_names = open(labels_path, "r").readlines()

# إنشاء مصفوفة لتغذية الصورة إلى النموذج
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# مسار الصورة (ضع مسار الصورة هنا)
image_path = "<IMAGE_PATH>"  # ضع المسار الكامل للصورة هنا
image = Image.open(image_path).convert("RGB")

# تغيير حجم الصورة إلى 224x224 وقصها من المنتصف
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# تحويل الصورة إلى مصفوفة
image_array = np.asarray(image)

# تطبيع الصورة لتتراوح القيم بين -1 و 1
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# إدخال الصورة إلى المصفوفة
data[0] = normalized_image_array

# استخدام النموذج لإجراء التوقع
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# طباعة النتائج
print(f"Predicted Class: {class_name.strip()}")
print(f"Confidence Score: {confidence_score:.2f}")
