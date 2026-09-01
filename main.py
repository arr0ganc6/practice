"""
=========================================================
Практическая работа

Тема:
Оптимизация скорости выполнения нейронных сетей
с помощью квантизации

Главный файл программы
=========================================================
"""

from model_manager import (
    train_all_models,
    load_all_models,
    check_saved_models
)

from quantization import (
    quantize_all_int8,
    quantize_all_int4
)

from benchmark import benchmark_models

from visualization import build_all_graphs

from utils import (
    print_header,
    print_success,
    print_error,
    print_warning,
    create_project_folders
)


# -------------------------------------------------------
# Полный цикл
# -------------------------------------------------------

def full_pipeline():

    print_header("ПОЛНЫЙ ЦИКЛ")

    create_project_folders()

    # --------------------------
    # Загрузка или обучение
    # --------------------------

    if check_saved_models():

        print_success(
            "Найдены сохраненные модели."
        )

        models = load_all_models()

    else:

        print_warning(
            "Сохраненные модели отсутствуют."
        )

        print("\nНачинается обучение...\n")

        models = train_all_models()

    # --------------------------
    # INT8
    # --------------------------

    print_header("INT8 КВАНТИЗАЦИЯ")

    int8_models = quantize_all_int8(models)

    # --------------------------
    # INT4
    # --------------------------

    print_header("INT4 КВАНТИЗАЦИЯ")

    int4_models = quantize_all_int4(models)

    # --------------------------
    # Benchmark
    # --------------------------

    print_header("ТЕСТИРОВАНИЕ")

    benchmark_models(

        original_models=models,

        int8_models=int8_models,

        int4_models=int4_models

    )

    # --------------------------
    # Графики
    # --------------------------

    print_header("ПОСТРОЕНИЕ ГРАФИКОВ")

    build_all_graphs()

    print_success(
        "Полный цикл завершён."
    )


# -------------------------------------------------------
# Меню
# -------------------------------------------------------

def menu():

    create_project_folders()

    models = None

    int8_models = None

    int4_models = None

    while True:

        print("\n")

        print("=" * 60)
        print("        ОПТИМИЗАЦИЯ CNN")
        print("=" * 60)

        print("1. Обучить модели")
        print("2. Загрузить модели")
        print("3. INT8 квантизация")
        print("4. INT4 квантизация")
        print("5. Benchmark")
        print("6. Построить графики")
        print("7. Полный цикл")
        print("0. Выход")

        print("=" * 60)

        choice = input(
            "Выберите пункт: "
        )

        # ------------------------------------

        if choice == "1":

            models = train_all_models()

        # ------------------------------------

        elif choice == "2":

            if check_saved_models():

                models = load_all_models()

            else:

                print_error(
                    "Сохранённые модели отсутствуют."
                )

        # ------------------------------------

        elif choice == "3":

            if models is None:

                print_warning(
                    "Сначала обучите или загрузите модели."
                )

                continue

            int8_models = quantize_all_int8(models)

        # ------------------------------------

        elif choice == "4":

            if models is None:

                print_warning(
                    "Сначала обучите или загрузите модели."
                )

                continue

            int4_models = quantize_all_int4(models)

        # ------------------------------------

        elif choice == "5":

            if models is None:

                print_warning(
                    "Нет моделей для тестирования."
                )

                continue

            benchmark_models(

                original_models=models,

                int8_models=int8_models,

                int4_models=int4_models

            )

        # ------------------------------------

        elif choice == "6":

            build_all_graphs()

        # ------------------------------------

        elif choice == "7":

            full_pipeline()

        # ------------------------------------

        elif choice == "0":

            print_success(
                "Работа завершена."
            )

            break

        # ------------------------------------

        else:

            print_error(
                "Неверный пункт меню."
            )


# -------------------------------------------------------
# Точка входа
# -------------------------------------------------------

if __name__ == "__main__":

    print_header(
        "Практическая работа"
    )

    print(
        "Оптимизация скорости выполнения\n"
        "нейронных сетей методом квантизации\n"
    )

    menu()