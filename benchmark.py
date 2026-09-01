import copy
import os
import time
import tempfile

import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from model_manager import get_test_loader

RESULTS_DIR = "results"

# Все измерения производительности выполняются только на CPU
BENCHMARK_DEVICE = torch.device("cpu")

torch.set_num_threads(
    os.cpu_count()
)

# -------------------------------------------------
# Создание папки результатов
# -------------------------------------------------

def create_results_folder():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------------------------------------
# Подготовка модели
# -------------------------------------------------

def prepare_model(model):
    """
    Создает независимую копию модели и переносит ее на CPU.
    """

    cpu_model = copy.deepcopy(model)

    cpu_model.to(BENCHMARK_DEVICE)
    cpu_model.eval()

    return cpu_model


# -------------------------------------------------
# Размер модели
# -------------------------------------------------

def get_model_size(model):

    with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pth"
    ) as file:

        torch.save(
            model,
            file.name
        )

        size = os.path.getsize(file.name)

    os.remove(file.name)

    return round(size / (1024 * 1024), 2)


# -------------------------------------------------
# Accuracy / Precision / Recall / F1
# -------------------------------------------------

def evaluate_model(model):

    model = prepare_model(model)

    loader = get_test_loader()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(BENCHMARK_DEVICE)
            labels = labels.to(BENCHMARK_DEVICE)

            outputs = model(images)

            prediction = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(prediction.numpy())

    return {

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


# -------------------------------------------------
# Скорость инференса
# -------------------------------------------------

def measure_inference_time(
        model,
        runs=300
):

    model = prepare_model(model)

    loader = get_test_loader()

    images, _ = next(iter(loader))

    images = images[:1]

    images = images.to(BENCHMARK_DEVICE)

    with torch.no_grad():

        # прогрев

        for _ in range(20):
            model(images)

        start = time.perf_counter()

        for _ in range(runs):
            model(images)

        finish = time.perf_counter()

    total = finish - start

    avg = total / runs

    fps = 1 / avg

    return round(avg * 1000, 3), round(fps, 2)


# -------------------------------------------------
# Полная проверка модели
# -------------------------------------------------

def benchmark_model(
        model,
        model_name,
        version
):

    metrics = evaluate_model(model)

    inference_time, fps = measure_inference_time(model)

    size = get_model_size(model)

    result = {

        "Model": model_name,

        "Version": version,

        "Accuracy": round(metrics["Accuracy"], 4),

        "Precision": round(metrics["Precision"], 4),

        "Recall": round(metrics["Recall"], 4),

        "F1": round(metrics["F1"], 4),

        "Time (ms)": inference_time,

        "FPS": fps,

        "Size (MB)": size
    }

    print()
    print("=" * 70)
    print(f"{model_name} ({version})")
    print("=" * 70)

    for key, value in result.items():

        if key in ("Model", "Version"):
            continue

        print(f"{key:<15}: {value}")

    return result


# -------------------------------------------------
# Сохранение результатов
# -------------------------------------------------

def save_results(results):

    create_results_folder()

    dataframe = pd.DataFrame(results)

    csv_path = os.path.join(
        RESULTS_DIR,
        "results.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "summary.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary = dataframe.pivot_table(

        index="Model",

        columns="Version",

        values=[
            "Accuracy",
            "Time (ms)",
            "FPS",
            "Size (MB)"
        ]
    )

    summary.to_csv(
        summary_path,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("Результаты сохранены")
    print("=" * 70)
    print(csv_path)
    print(summary_path)

    return dataframe


# -------------------------------------------------
# Лучшие результаты
# -------------------------------------------------

def print_best_results(dataframe):

    print()
    print("=" * 70)
    print("ЛУЧШИЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    best_accuracy = dataframe.loc[
        dataframe["Accuracy"].idxmax()
    ]

    fastest = dataframe.loc[
        dataframe["Time (ms)"].idxmin()
    ]

    smallest = dataframe.loc[
        dataframe["Size (MB)"].idxmin()
    ]

    print("\nЛучшая Accuracy\n")
    print(best_accuracy)

    print("\nСамая быстрая\n")
    print(fastest)

    print("\nСамая компактная\n")
    print(smallest)


# -------------------------------------------------
# Тестирование всех моделей
# -------------------------------------------------

def benchmark_models(

        original_models,
        int8_models,
        int4_models

):

    print()
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ (CPU)")
    print("=" * 70)

    results = []

    groups = [

        ("Original", original_models),

        ("INT8", int8_models),

        ("INT4", int4_models)

    ]

    for version, models in groups:

        if models is None:
            continue

        print(f"\n>>> {version}\n")

        for model_name, model in models.items():

            results.append(

                benchmark_model(

                    model,

                    model_name,

                    version

                )

            )

    dataframe = save_results(results)

    print_best_results(dataframe)

    print()
    print("=" * 70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 70)

    print(dataframe.to_string(index=False))

    return dataframe


# -------------------------------------------------
# Загрузка результатов
# -------------------------------------------------

def load_results():

    path = os.path.join(
        RESULTS_DIR,
        "results.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError("Файл результатов не найден.")

    return pd.read_csv(path)


# -------------------------------------------------
# Данные для графиков
# -------------------------------------------------

def get_plot_data():

    dataframe = load_results()

    return {

        "models": dataframe["Model"],

        "versions": dataframe["Version"],

        "accuracy": dataframe["Accuracy"],

        "time": dataframe["Time (ms)"],

        "fps": dataframe["FPS"],

        "size": dataframe["Size (MB)"]

    }


# -------------------------------------------------
# Проверка
# -------------------------------------------------

if __name__ == "__main__":

    print("benchmark.py используется как модуль.")