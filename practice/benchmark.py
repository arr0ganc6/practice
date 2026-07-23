
import os
import time
import tempfile

import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from model_manager import (
    DEVICE,
    get_test_loader
)

RESULTS_DIR = "results"

def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

# -------------------------------------------------
# Создание папки результатов
# -------------------------------------------------

def create_results_folder():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


# -------------------------------------------------
# Размер модели
# -------------------------------------------------

def get_model_size(model):

    with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pth"
    ) as temp_file:
        torch.save(
            model.state_dict(),
            temp_file.name
        )

        size_mb = (
            os.path.getsize(temp_file.name)
            / (1024 * 1024)
        )

    os.remove(temp_file.name)

    return round(size_mb, 2)


# -------------------------------------------------
# Accuracy / Precision / Recall / F1
# -------------------------------------------------

def evaluate_model(model):

    model.eval()

    loader = get_test_loader()

    # выбираем устройство по модели
    device = get_model_device(model)

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    metrics = {

        "Accuracy":

            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":

            precision_score(

                y_true,

                y_pred,

                average="macro",

                zero_division=0

            ),

        "Recall":

            recall_score(

                y_true,

                y_pred,

                average="macro",

                zero_division=0

            ),

        "F1":

            f1_score(

                y_true,

                y_pred,

                average="macro",

                zero_division=0

            )

    }

    return metrics


# -------------------------------------------------
# Скорость инференса
# -------------------------------------------------

def measure_inference_time(

        model,

        runs=300

):

    model.eval()

    loader = get_test_loader()

    images, _ = next(iter(loader))

    device = get_model_device(model)

    images = images.to(device)

    batch_size = images.shape[0]

    with torch.no_grad():

        # прогрев

        for _ in range(20):

            _ = model(images)

        if torch.cuda.is_available():

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

        start = time.perf_counter()

        for _ in range(runs):

            _ = model(images)

        if torch.cuda.is_available():

            torch.cuda.synchronize()

        finish = time.perf_counter()

    total_time = finish - start

    average_time = total_time / runs

    fps = batch_size / average_time

    return (

        round(average_time * 1000, 3),

        round(fps, 2)

    )


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

        "Accuracy":

            round(
                metrics["Accuracy"],
                4
            ),

        "Precision":

            round(
                metrics["Precision"],
                4
            ),

        "Recall":

            round(
                metrics["Recall"],
                4
            ),

        "F1":

            round(
                metrics["F1"],
                4
            ),

        "Time (ms)": inference_time,

        "FPS": fps,

        "Size (MB)": size

    }

    print("\n")

    print("=" * 70)

    print(

        f"{model_name} ({version})"

    )

    print("=" * 70)

    for key, value in result.items():

        if key in [

            "Model",

            "Version"

        ]:

            continue

        print(

            f"{key:<15}: {value}"

        )

    return result
# -------------------------------------------------
# Сохранение результатов
# -------------------------------------------------

def save_results(results):

    create_results_folder()

    df = pd.DataFrame(results)

    csv_path = os.path.join(
        RESULTS_DIR,
        "results.csv"
    )

    #excel_path = os.path.join(
    #    RESULTS_DIR,
    #    "results.xlsx"
    #)

    summary_path = os.path.join(
        RESULTS_DIR,
        "summary.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    #df.to_excel(
    #    excel_path,
    #    index=False
    #)

    summary = df.pivot_table(

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

    print("\n")

    print("=" * 70)

    print("Результаты сохранены")

    print("=" * 70)

    print(csv_path)

    #print(excel_path)

    print(summary_path)

    return df


# -------------------------------------------------
# Поиск лучших моделей
# -------------------------------------------------

def print_best_results(df):

    print("\n")

    print("=" * 70)
    print("ЛУЧШИЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    best_accuracy = df.loc[
        df["Accuracy"].idxmax()
    ]

    fastest = df.loc[
        df["Time (ms)"].idxmin()
    ]

    smallest = df.loc[
        df["Size (MB)"].idxmin()
    ]

    print("\nЛучшая Accuracy")

    print(best_accuracy)

    print("\nСамая быстрая")

    print(fastest)

    print("\nСамая компактная")

    print(smallest)


# -------------------------------------------------
# Главное тестирование
# -------------------------------------------------

def benchmark_models(

        original_models,

        int8_models,

        int4_models

):

    results = []

    model_groups = [

        (

            "Original",

            original_models

        ),

        (

            "INT8",

            int8_models

        ),

        (

            "INT4",

            int4_models

        )

    ]

    print("\n")

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
    print("=" * 70)

    for version, models in model_groups:

        if models is None:
            continue

        print(f"\n>>> {version}\n")

        for model_name, model in models.items():

            result = benchmark_model(

                model,

                model_name,

                version

            )

            results.append(result)

    dataframe = save_results(results)

    print_best_results(dataframe)

    print("\n")

    print("=" * 70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 70)

    print(dataframe.to_string(index=False))

    return dataframe


# -------------------------------------------------
# Загрузка сохранённых результатов
# -------------------------------------------------

def load_results():

    path = os.path.join(

        RESULTS_DIR,

        "results.csv"

    )

    if not os.path.exists(path):

        raise FileNotFoundError(

            "Файл результатов не найден."

        )

    return pd.read_csv(path)


# -------------------------------------------------
# Получение данных для графиков
# -------------------------------------------------

def get_plot_data():

    dataframe = load_results()

    return {

        "models":

            dataframe["Model"],

        "versions":

            dataframe["Version"],

        "accuracy":

            dataframe["Accuracy"],

        "time":

            dataframe["Time (ms)"],

        "fps":

            dataframe["FPS"],

        "size":

            dataframe["Size (MB)"]

    }


# -------------------------------------------------
# Проверка
# -------------------------------------------------

if __name__ == "__main__":

    print(

        "benchmark.py "

        "используется "

        "как модуль."

    )