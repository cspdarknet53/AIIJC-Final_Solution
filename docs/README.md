# Классификация дорожных знаков

## Установка
1. Склонируйте репозиторий:
   ```shell
   git clone https://github.com/2gis/signs_classification_aij2021.git 
   cd signs_classification_aij2021
   ```
2. Установите необходимые зависимости:
   ```shell
    pip install -r requirements.txt
   ```
3. Загрузите датасет. По умолчанию считается, что датасет будет находиться в папке `data/` в
этом же проекте. Если это не так, то поменяйте `DEFAULT_DATA_PATH` из `pipeline/constants.py`
на нужный путь.
4. Скачайте веса с https://drive.google.com/file/d/1rPQHBvp8w_F9Nrtbr1p6lAx__zCTMXAu/view?usp=sharing и поместите их в ./Web app/app
   
## Обучение нейросети

По умолчанию эксперименты сохраняются в папке `experiments/`. Если хочется сделать по-другому,
то нужно поменять `DEFAULT_EXPERIMENTS_SAVE_PATH` из `pipeline/constants.py` на нужный путь.
Веса лучшей модели сохраняются как `experiments/experiment_name/best.pth`.
### Входные параметры
- **exp_name** - название эксперимента (это папка будет создана в папке `experiments/`)
- **n_epochs** - количество эпох для тренировки
- **model_name** - имя энкодера сети. Доступные энкодеры описаны в словаре `ENCODERS` из
`pipeline/models.py`
- **batch_size** - размер батча
- **device** - устройство для вычислений

### Запуск

   ```shell
   python -m pipeline.train \
       --exp_name baseline \
       --n_epochs 50 \
       --model_name resnet18 \
       --batch_size 8 \
       --device cuda:0
   ```

## Тестовый скрипт

Скрипт вычисляет ответы для тестовых изображений и сохраняет результат в папку эксперимента
в файл с названием `submit.csv`.
### Входные параметры
- **exp_name** - имя эксперимента, из которого тестируется модель
- **model_name** - имя энкодера модели, которая будет тестироваться
- **batch_size** - размер батча
- **device** - устройство для вычислений

### Запуск
   ```shell
   python -m pipeline.generate_submission \
       --exp_name baseline \
       --model_name resnet18 \
       --batch_size 8 \
       --device cuda:0
   ```
