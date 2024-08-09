import random
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
 

# Load the pre-trained model for image classification
model = load_model('/Users/kirillrabdel/Projectt/autism_detection_model.keras')
# функция предобработки изоб.
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))   
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0   
    return img_array

def predict_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5:
        return "Изображение относится к категории 'Autism'."
    else:
        return "Изображение относится к категории 'NonAutism'."

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет, я бот который может класифицировать ASD на основе лицевых черт.Отправь мне фото которое ты хочешь проанализировать !')

async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Чем могу помочь?')

 
async def run_bot_image(update: Update, context: CallbackContext) -> None:
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    file_path = 'image.jpg'
    await file.download_to_drive(file_path)
    
  
    result = predict_image(file_path)
    
    # Send back the prediction result
    await update.message.reply_text(result)
    
    # Optionally remove the image file after processing
    os.remove(file_path)

def main():
    application = Application.builder().token("7372571284:AAGTa145zzk-8Lo3511fFMu--XrDRL0Q1LU").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, run_bot_image))
    application.run_polling()

if __name__ == "__main__":
    main()
