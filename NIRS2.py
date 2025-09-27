import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Загрузка данных
df = pd.read_csv('/content/drive/MyDrive/DSL-StrongPasswordData.csv')
print("Размер датасета:", df.shape)
print("Уникальные пользователи:", len(df['subject'].unique()))

# 1. ПОДГОТОВКА ДАННЫХ ДЛЯ БИНАРНОЙ КЛАССИФИКАЦИИ
# Целевой пользователь vs все остальные
target_user = 's002'
df['is_target'] = (df['subject'] == target_user).astype(int)

print("Распределение классов:")
print(df['is_target'].value_counts())
print(f"Доля целевого пользователя: {df['is_target'].mean():.3f}")

# 2. БАЛАНСИРОВКА ДАННЫХ
# Чтобы избежать дисбаланса, возьмем равное количество примеров от каждого класса
target_data = df[df['is_target'] == 1]
non_target_data = df[df['is_target'] == 0].sample(n=len(target_data), random_state=42)

balanced_df = pd.concat([target_data, non_target_data])
print(f"\nСбалансированный датасет: {balanced_df.shape}")

# 3. ПОДГОТОВКА ПРИЗНАКОВ И МЕТОК
features = balanced_df.drop(['subject', 'sessionIndex', 'rep', 'is_target'], axis=1)
labels = balanced_df['is_target']

# Масштабирование признаков
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. РАЗДЕЛЕНИЕ ДАННЫХ С СОХРАНЕНИЕМ ГРУПП (ПОЛЬЗОВАТЕЛЕЙ)
# Важно: разделяем по пользователям, а не случайным образом
unique_subjects = balanced_df['subject'].unique()
target_subjects = [target_user]
non_target_subjects = [s for s in unique_subjects if s != target_user]

# Разделяем нецелевых пользователей на train/test
np.random.seed(42)
train_non_target = np.random.choice(non_target_subjects, 
                                   size=int(0.7 * len(non_target_subjects)), 
                                   replace=False)
test_non_target = [s for s in non_target_subjects if s not in train_non_target]

# Создаем маски для разделения
train_mask = balanced_df['subject'].isin(list(train_non_target) + [target_user])
test_mask = balanced_df['subject'].isin(list(test_non_target) + [target_user])

X_train = features_scaled[train_mask]
X_test = features_scaled[test_mask]
y_train = labels[train_mask]
y_test = labels[test_mask]

print(f"\nРазделение данных:")
print(f"Обучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка: {X_test.shape[0]} примеров")
print(f"Целевой класс в train: {y_train.mean():.3f}")
print(f"Целевой класс в test: {y_test.mean():.3f}")

# 5. МОДЕЛИ БИНАРНОЙ КЛАССИФИКАЦИИ
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
}

# 6. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ
results = {}

for name, model in models.items():
    print(f"\n{'-'*50}")
    print(f"Обучение {name}")
    print('-'*50)
    
    # Обучение модели
    model.fit(X_train, y_train)
    
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Вероятность класса 1 (целевой пользователь)
    
    # Оценка качества
    print(classification_report(y_test, y_pred, target_names=['Нецелевой', 'Целевой']))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок:")
    print(cm)
    
    # ROC-AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {auc_score:.3f}")
    
    # Сохранение результатов
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'auc': auc_score,
        'confusion_matrix': cm
    }

# 7. ВЫБОР ЛУЧШЕЙ МОДЕЛИ И АНАЛИЗ
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']

print(f"\n{'='*60}")
print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
print(f"ROC-AUC: {results[best_model_name]['auc']:.3f}")
print('='*60)

# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
plt.figure(figsize=(15, 5))

# Матрица ошибок лучшей модели
plt.subplot(1, 3, 1)
cm = results[best_model_name]['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Нецелевой', 'Целевой'],
            yticklabels=['Нецелевой', 'Целевой'])
plt.title(f'Матрица ошибок ({best_model_name})')
plt.ylabel('Фактический класс')
plt.xlabel('Предсказанный класс')

# Важность признаков (для Random Forest)
plt.subplot(1, 3, 2)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Важность признаков (топ-15)')
    plt.xlabel('Важность')
else:
    plt.text(0.5, 0.5, 'Недоступно для этой модели', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Важность признаков')

# Сравнение ROC-AUC всех моделей
plt.subplot(1, 3, 3)
model_names = list(results.keys())
auc_scores = [results[name]['auc'] for name in model_names]
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = plt.bar(model_names, auc_scores, color=colors)
plt.ylim(0, 1)
plt.title('Сравнение ROC-AUC моделей')
plt.ylabel('ROC-AUC')

# Добавляем значения на столбцы
for bar, score in zip(bars, auc_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 9. ГЛУБОКИЙ АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ
print(f"\nДЕТАЛЬНЫЙ АНАЛИЗ {best_model_name}:")

# Анализ ошибочных предсказаний
y_test_values = y_test.values
best_predictions = results[best_model_name]['y_pred']
best_probabilities = results[best_model_name]['y_pred_proba']

# Примеры с наибольшей неопределенностью
uncertain_threshold = 0.4  # Порог неопределенности
uncertain_mask = (best_probabilities > uncertain_threshold) & (best_probabilities < (1 - uncertain_threshold))
uncertain_count = uncertain_mask.sum()

print(f"Примеров с неопределенностью (0.4 < p < 0.6): {uncertain_count}/{len(y_test)}")

# Анализ по пользователям (для тестовых данных)
test_subjects = balanced_df['subject'][test_mask].values
results_df = pd.DataFrame({
    'subject': test_subjects,
    'true_label': y_test_values,
    'predicted_label': best_predictions,
    'probability': best_probabilities
})

# Статистика по пользователям
user_stats = results_df.groupby('subject').agg({
    'true_label': 'first',
    'predicted_label': 'mean',
    'probability': ['mean', 'std', 'count']
}).round(3)

print("\nСтатистика по пользователям в тестовой выборке:")
print(user_stats)

# 10. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': features.columns.tolist(),
    'target_user': target_user,
    'performance': {
        'auc': results[best_model_name]['auc'],
        'confusion_matrix': results[best_model_name]['confusion_matrix']
    }
}

joblib.dump(model_artifacts, '/content/drive/MyDrive/best_keystroke_binary_model.pkl')
print(f"\nЛучшая модель сохранена: /content/drive/MyDrive/best_keystroke_binary_model.pkl")

# 11. ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ
def predict_user(model_path, new_keystroke_data):
    """
    Функция для предсказания принадлежности пользователя
    
    Parameters:
    - model_path: путь к сохраненной модели
    - new_keystroke_data: данные ввода пароля (DataFrame или array)
    
    Returns:
    - prediction: 1 если целевой пользователь, 0 если нет
    - probability: вероятность принадлежности целевому пользователю
    - confidence: уровень уверенности (высокий/средний/низкий)
    """
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    scaler = artifacts['scaler']
    
    # Масштабирование новых данных
    if hasattr(new_keystroke_data, 'values'):
        new_data_scaled = scaler.transform(new_keystroke_data.values)
    else:
        new_data_scaled = scaler.transform(new_keystroke_data)
    
    # Предсказание
    probability = model.predict_proba(new_data_scaled)[0, 1]
    prediction = 1 if probability > 0.5 else 0
    
    # Уровень уверенности
    if probability > 0.8 or probability < 0.2:
        confidence = "высокий"
    elif probability > 0.6 or probability < 0.4:
        confidence = "средний"
    else:
        confidence = "низкий"
    
    return prediction, probability, confidence

print(f"\n{'='*60}")
print("ФУНКЦИЯ ДЛЯ ИСПОЛЬЗОВАНИЯ В ПРОДУКШЕНЕ")
print('='*60)
print("""
# Пример использования:
# prediction, prob, confidence = predict_user(
#     '/content/drive/MyDrive/best_keystroke_binary_model.pkl',
#     new_keystroke_data
# )
""")