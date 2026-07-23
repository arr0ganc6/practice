"""
visualization.py

Построение графиков результатов benchmark

Графики:

- Accuracy
- Time
- FPS
- Size

Все графики сохраняются в папку results
"""

import os

import pandas as pd

import matplotlib.pyplot as plt


# =====================================================
# Папка результатов
# =====================================================

RESULTS_DIR = "results"

CSV_FILE = os.path.join(
    RESULTS_DIR,
    "results.csv"
)


# =====================================================
# Загрузка результатов
# =====================================================

def load_results():

    if not os.path.exists(CSV_FILE):

        raise FileNotFoundError(

            "Файл results.csv не найден."

        )

    dataframe = pd.read_csv(CSV_FILE)

    return dataframe


# =====================================================
# Сохранение графика
# =====================================================

def save_plot(name):

    path = os.path.join(

        RESULTS_DIR,

        name

    )

    plt.tight_layout()

    plt.savefig(

        path,

        dpi=300

    )

    plt.close()


# =====================================================
# Accuracy
# =====================================================

def plot_accuracy(dataframe):

    plt.figure(figsize=(12,6))

    labels = (

        dataframe["Model"]

        + "\n"

        + dataframe["Version"]

    )

    plt.bar(

        labels,

        dataframe["Accuracy"]

    )

    plt.title(

        "Accuracy"

    )

    plt.ylabel(

        "Accuracy"

    )

    plt.xticks(

        rotation=25

    )

    plt.grid(

        axis="y"

    )

    save_plot(

        "accuracy.png"

    )

    print(

        "accuracy.png сохранён"

    )


# =====================================================
# Time
# =====================================================

def plot_time(dataframe):

    plt.figure(figsize=(12,6))

    labels = (

        dataframe["Model"]

        + "\n"

        + dataframe["Version"]

    )

    plt.bar(

        labels,

        dataframe["Time (ms)"]

    )

    plt.title(

        "Inference Time"

    )

    plt.ylabel(

        "Milliseconds"

    )

    plt.xticks(

        rotation=25

    )

    plt.grid(

        axis="y"

    )

    save_plot(

        "time.png"

    )

    print(

        "time.png сохранён"

    )
# =====================================================
# FPS
# =====================================================

def plot_fps(dataframe):

    plt.figure(figsize=(12, 6))

    labels = (
        dataframe["Model"]
        + "\n"
        + dataframe["Version"]
    )

    plt.bar(

        labels,

        dataframe["FPS"]

    )

    plt.title(

        "FPS"

    )

    plt.ylabel(

        "Frames per second"

    )

    plt.xticks(

        rotation=25

    )

    plt.grid(

        axis="y"

    )

    save_plot(

        "fps.png"

    )

    print(

        "fps.png сохранён"

    )


# =====================================================
# Размер модели
# =====================================================

def plot_size(dataframe):

    plt.figure(figsize=(12, 6))

    labels = (
        dataframe["Model"]
        + "\n"
        + dataframe["Version"]
    )

    plt.bar(

        labels,

        dataframe["Size (MB)"]

    )

    plt.title(

        "Model Size"

    )

    plt.ylabel(

        "MB"

    )

    plt.xticks(

        rotation=25

    )

    plt.grid(

        axis="y"

    )

    save_plot(

        "size.png"

    )

    print(

        "size.png сохранён"

    )


# =====================================================
# Построение всех графиков
# =====================================================

def build_all_graphs():

    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 60)

    dataframe = load_results()

    plot_accuracy(
        dataframe
    )

    plot_time(
        dataframe
    )

    plot_fps(
        dataframe
    )

    plot_size(
        dataframe
    )

    print("\nВсе графики успешно сохранены.")

    print(
        os.path.join(
            RESULTS_DIR,
            "accuracy.png"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "time.png"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "fps.png"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "size.png"
        )
    )


# =====================================================
# Проверка
# =====================================================

if __name__ == "__main__":

    build_all_graphs()