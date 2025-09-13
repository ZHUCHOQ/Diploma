import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
data = pd.read_csv('/content/drive/MyDrive/DSL-StrongPasswordData.csv')  # Замените на путь к вашему файлу

# Выбор субъекта для метки "свой" (здесь s002)
own_subject = 's002'
data['label'] = data['subject'].apply(lambda x: 1 if x == own_subject else 0)

# Создание сбалансированного датасета
own_data = data[data['label'] == 1]
other_data = data[data['label'] == 0]

# Исправленная выборка: по 8 случайных записей от каждого чужого субъекта
other_sampled = other_data.groupby('subject', group_keys=False).apply(lambda x: x.sample(8, random_state=42))

# Объединение данных
balanced_data = pd.concat([own_data, other_sampled], ignore_index=True)

# Проверка баланса классов
print("Распределение классов:")
print(balanced_data['label'].value_counts())

# Разделение на признаки и метки
X = balanced_data.drop(['subject', 'sessionIndex', 'rep', 'label'], axis=1)
y = balanced_data['label']

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Преобразование в 3D-форму для LSTM [samples, timesteps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y
)

# Упрощенная модель с меньшей регуляризацией
model = Sequential()
model.add(LSTM(32, 
               input_shape=(X_train.shape[1], X_train.shape[2]),
               kernel_regularizer=l2(0.001),
               return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Ранняя остановка
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение модели
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Предсказания
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Детальный отчет
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Визуализация результатов
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# График точности
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# График потерь
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax3)
ax3.set_title('Confusion Matrix')

# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
ax4.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
ax4.plot([0, 1], [0, 1], 'k--')  # Случайный классификатор
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('Receiver Operating Characteristic')
ax4.legend(loc="lower right")

plt.tight_layout()
plt.show()

# Анализ важности признаков (на основе весов модели)
# Получаем веса из первого слоя LSTM
lstm_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(lstm_weights), axis=1)

# Создаем DataFrame для визуализации
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Визуализация важности признаков
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(15))
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.show()