# FastSpeech

В этом проекте реализация двух версия FastSpeech для генерации аудио по тексте.

# Настройка окружения

Первым делом нужно настроить окружение python3.10 из файлы requirements.txt

Далее необходимо скачать файлы длы обучения FastSpeech и FastSpeech2 скпритами
`setup.sh` и `setup2.sh` соответственно.


# Обучение

Для обучения выберете конфиг для нужной модели:

## FastSpeech 1

```bash
python3 train.py -c src/configs/fastspeech1/train.json
```

## FastSpeech 2
```bash
python3 train.py -c src/configs/fastspeech2/train.json
```

# Тест

Этот скрипт сгенерирует тестовые аудио для трех валидационных текстов:

- A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest
- Massachusetts Institute of Technology may be best known for its math, science and engineering education
- Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space

Чекпоинт можно скачать по этой ссылке

https://drive.google.com/file/d/1traOcYCVQ8mrQcimnHNQdPGprvln2bJF/view?usp=sharing

```bash
python3 test.py -c src/configs/fastspeech2/test.json -r model_best.pth
```