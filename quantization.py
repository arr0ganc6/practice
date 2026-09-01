"""
quantization.py

Посттренировочная квантизация моделей
(PTQ FX Graph Mode)

INT8 - реальная квантизация
INT4 - временная заглушка (используется INT8)

"""

import copy
import warnings

import torch

from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping
)

from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx
)

from model_manager import (
    get_calibration_loader
)

warnings.filterwarnings("ignore")


# =====================================================
# Калибровка модели
# =====================================================

def calibrate_model(
        prepared_model,
        batches=100
):

    loader = get_calibration_loader()

    prepared_model.eval()

    with torch.no_grad():

        for index, (images, _) in enumerate(loader):

            if index >= batches:
                break

            prepared_model(images.cpu())


# =====================================================
# INT8 FX Quantization
# =====================================================

def quantize_int8(model):

    print("Квантизация INT8...")

    model = copy.deepcopy(model)

    model.cpu()

    model.eval()

    qconfig = get_default_qconfig(
        "fbgemm"
    )

    qconfig_mapping = (
        QConfigMapping()
        .set_global(qconfig)
    )

    example_inputs = (
        torch.randn(
            1,
            3,
            224,
            224
        ),
    )

    prepared_model = prepare_fx(

        model,

        qconfig_mapping,

        example_inputs

    )

    calibrate_model(prepared_model)

    quantized_model = convert_fx(
        prepared_model
    )

    quantized_model.eval()

    return quantized_model


# =====================================================
# INT4
# =====================================================

def quantize_int4(model):

    print(
        "INT4 недоступен."
    )

    print(
        "Используется INT8."
    )

    return quantize_int8(model)
# =====================================================
# Квантизация всех моделей (INT8)
# =====================================================

def quantize_all_int8(models):

    print("\n" + "=" * 60)
    print("КВАНТИЗАЦИЯ INT8")
    print("=" * 60)

    quantized_models = {}

    for name, model in models.items():

        print(f"\n{name}")

        try:

            quantized_models[name] = quantize_int8(model)

            print("✓ Успешно")

        except Exception as error:

            print(f"Ошибка: {error}")

    return quantized_models


# =====================================================
# Квантизация всех моделей (INT4)
# =====================================================

def quantize_all_int4(models):

    print("\n" + "=" * 60)
    print("КВАНТИЗАЦИЯ INT4")
    print("=" * 60)

    quantized_models = {}

    for name, model in models.items():

        print(f"\n{name}")

        try:

            quantized_models[name] = quantize_int4(model)

            print("✓ Успешно")

        except Exception as error:

            print(f"Ошибка: {error}")

    return quantized_models


# =====================================================
# Информация о модели
# =====================================================

def print_model_info(model, title):

    params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(f"Всего параметров : {params:,}")
    print(f"Обучаемых        : {trainable:,}")


# =====================================================
# Проверка работы
# =====================================================

if __name__ == "__main__":

    print(
        "Модуль quantization.py "
        "предназначен для импорта."
    )