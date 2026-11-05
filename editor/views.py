"""Загрузка файлов, состояние сессии, параметры и вывод результатов."""
import time
import uuid
from decimal import Decimal, InvalidOperation

import cv2
import numpy as np
from PIL import Image
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.utils.safestring import mark_safe

from .catalog import METHOD_GROUPS, OPERATIONS, VARIANT_HELP
from .processing import calculate_metrics, decode_image, demo_image, encode_png, histogram_svg, process_image

QUALITY_OPERATIONS = {"smooth", "median", "sharpen", "frequency"}


def save_image(state, image):
    name = f"{state['folder']}/{uuid.uuid4().hex}.png"
    path = settings.MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(image))
    return name


def read_image(name):
    with Image.open(settings.MEDIA_ROOT / name) as image:
        return np.asarray(image.convert("RGB")).copy()


def parse_checkbox(field, value):
    return value is True or str(value) in ("on", "true", "1")


def parse_choice(field, value):
    if str(value) not in [option["value"] for option in field["options"]]:
        raise ValueError(f"Недопустимое значение «{field['label']}».")
    return int(value) if field["id"] in ("size", "channel") else str(value)


def parse_number(field, value):
    try:
        number = Decimal(str(value))
        minimum, maximum, step = (Decimal(str(field[name])) for name in ("min", "max", "step"))
        if not number.is_finite() or not minimum <= number <= maximum or (number - minimum) % step:
            raise ValueError
    except (InvalidOperation, ValueError):
        raise ValueError(f"Параметр «{field['label']}»: от {field['min']} до {field['max']}, шаг {field['step']}.")
    return int(number) if step == int(step) else float(number)


PARAMETER_PARSERS = {
    "checkbox": parse_checkbox,
    "select": parse_choice,
    "number": parse_number,
}


def parse_parameters(operation_id, values):
    parameters = {}
    for field in OPERATIONS[operation_id]["parameters"]:
        parser = PARAMETER_PARSERS[field["type"]]
        value = values.get(field["id"], field["default"])
        parameters[field["id"]] = parser(field, value)
    return parameters



def describe_parameters(operation_id, parameters):
    parts = []
    for field in OPERATIONS[operation_id]["parameters"]:
        key, value = field["id"], parameters[field["id"]]
        if operation_id == "noise" and ((key == "sigma" and parameters["type"] == "salt") or (key == "amount" and parameters["type"] != "salt")):
            continue
        if field["type"] == "select":
            value = next(choice["label"] for choice in field["options"] if choice["value"] == str(value))
        parts.append(f"{field['label']}: {value}")
    return " · ".join(parts) or "Без параметров"


def current_result(state):
    return next((entry for entry in state["results"] if entry["id"] == state.get("selected")), None)


def result_metrics(state, result):
    if result and result["operation"] in QUALITY_OPERATIONS:
        return calculate_metrics(read_image(state.get("reference") or state["original"]), read_image(result["image"]))
    return None


def handle_upload(request, state):
    action = request.POST.get('action')
    image = demo_image() if action == 'demo' else decode_image(request.FILES.get('image'))
    name = save_image(state, image)
    state.update(original=name, current=name, reference=None, results=[], selected=None, filename='Учебный пример' if action == 'demo' else request.FILES['image'].name, input_name='Исходное изображение')
    message = 'Изображение загружено.'
    return message


def handle_reference(request, state):
    if not state.get('original'):
        raise ValueError('Сначала загрузите основное изображение.')
    image = decode_image(request.FILES.get('reference'))
    if image.shape != read_image(state['original']).shape:
        raise ValueError('После уменьшения размеры эталона должны совпадать с основным изображением.')
    state['reference'] = save_image(state, image)
    state['reference_name'] = request.FILES['reference'].name
    message = 'Эталон загружен. Метрики пересчитаны.'
    return message


def handle_reset_reference(request, state):
    state['reference'] = None
    message = 'Эталон — первоначально загруженное изображение.'
    return message


def handle_processing(request, state):
    action = request.POST.get('action')
    operation_id = state['operation']
    values = request.POST.copy()
    for field in OPERATIONS[operation_id]['parameters']:
        if field['type'] == 'checkbox':
            values[field['id']] = request.POST.get(field['id'], '')
    parameters = parse_parameters(operation_id, values)
    state['parameters'][operation_id] = parameters
    if action == 'process':
        if not state.get('current'):
            raise ValueError('Сначала загрузите изображение.')
        start = time.perf_counter()
        result, extras, note = process_image(read_image(state['current']), operation_id, parameters)
        entry = {'id': uuid.uuid4().hex, 'operation': operation_id, 'name': OPERATIONS[operation_id]['name'], 'parameters': parameters, 'label': describe_parameters(operation_id, parameters), 'before': state['current'], 'image': save_image(state, result), 'note': note, 'extras': [{'name': title, 'image': save_image(state, pixels)} for title, pixels in extras]}
        state['results'] = (state['results'] + [entry])[-12:]
        state['selected'] = entry['id']
        message = f'Готово за {(time.perf_counter() - start) * 1000:.0f} мс. Обработка выполнена в Python.'
    else:
        message = 'Параметры и пояснения обновлены.'
    return message


def handle_use_result(request, state):
    selected = current_result(state)
    if not selected:
        raise ValueError('Сначала выполните обработку.')
    if selected['operation'] in ('space', 'fourier'):
        raise ValueError('Выберите обычное RGB-изображение, а не визуализацию компонент или спектра.')
    state['current'], state['input_name'] = (selected['image'], selected['name'])
    message = 'Результат выбран как вход. Выберите следующую операцию. Эталон сохранён.'
    return message


def handle_original(request, state):
    state['current'], state['input_name'] = (state.get('original'), 'Исходное изображение')
    message = 'Первоначальное изображение выбрано как вход.'
    return message


def handle_select_result(request, state):
    selected = next((entry for entry in state['results'] if entry['id'] == request.POST.get('result_id')), None)
    if selected:
        state['selected'], state['current'] = (selected['id'], selected['before'])
        state['input_name'] = 'Изображение до выбранной обработки'
    return ''


def handle_clear_results(request, state):
    state.update(results=[], selected=None, current=state.get('original'), input_name='Исходное изображение')
    message = 'Результаты очищены.'
    return message


ACTION_HANDLERS = {
    "upload": handle_upload,
    "demo": handle_upload,
    "reference": handle_reference,
    "reset_reference": handle_reset_reference,
    "configure": handle_processing,
    "process": handle_processing,
    "use_result": handle_use_result,
    "original": handle_original,
    "select": handle_select_result,
    "clear": handle_clear_results,
}


def index(request):
    state = request.session.get("lab", {"folder": uuid.uuid4().hex, "results": [], "parameters": {}})
    operation_id = request.POST.get("operation", request.GET.get("operation", state.get("operation", "space")))
    if operation_id not in OPERATIONS:
        operation_id = "space"
    state["operation"] = operation_id
    message, error = state.pop("message", ""), False

    if request.method == "POST":
        action = request.POST.get("action")
        try:
            handler = ACTION_HANDLERS.get(action)
            if handler is None:
                raise ValueError("Неизвестное действие.")
            message = handler(request, state)
            state["message"] = message
            request.session["lab"] = state
            return redirect(f"/?operation={operation_id}")
        except (ValueError, OSError, cv2.error) as exception:
            message, error = str(exception), True

    parameters = state["parameters"].get(operation_id, {field["id"]: field["default"] for field in OPERATIONS[operation_id]["parameters"]})
    fields, help_texts = [], []
    for source in OPERATIONS[operation_id]["parameters"]:
        field = dict(source)
        key = field["id"]
        field["value"] = str(parameters[key]) if field["type"] != "checkbox" else bool(parameters[key])
        field["help"] = VARIANT_HELP.get(key, {}).get(str(parameters[key]), "")
        if field["help"]:
            help_texts.append(field["help"])
        field["inactive"] = (operation_id == "noise" and ((key == "sigma" and parameters.get("type") == "salt") or (key == "amount" and parameters.get("type") != "salt"))) or (operation_id == "smooth" and key == "sigma" and parameters.get("kind") == "mean")
        fields.append(field)

    selected = current_result(state)
    current = read_image(state["current"]) if state.get("current") else None
    result = read_image(selected["image"]) if selected else None
    for entry in state["results"]:
        entry["metrics"] = result_metrics(state, entry)
        entry["selected"] = entry["id"] == state.get("selected")
    groups = [{"title": title, "methods": [{"id": key, "name": OPERATIONS[key]["name"]} for key in methods]} for _, title, methods in METHOD_GROUPS]
    request.session["lab"] = state
    context = {"state": state, "operation_id": operation_id, "operation": OPERATIONS[operation_id], "groups": groups,
               "fields": fields, "help_texts": help_texts, "selected": selected,
               "can_use": bool(selected and selected["operation"] not in ("space", "fourier")),
               "metrics": result_metrics(state, selected), "message": message, "error": error,
               "media": settings.MEDIA_URL, "current_size": f"{current.shape[1]} × {current.shape[0]} px" if current is not None else "",
               "result_size": f"{result.shape[1]} × {result.shape[0]} px" if result is not None else "",
               "hist_before": mark_safe(histogram_svg(current)) if current is not None else "",
               "hist_after": mark_safe(histogram_svg(result)) if result is not None else ""}
    return render(request, "index.html", context)


def download_result(request, result_id):
    state = request.session.get("lab", {})
    result = next((entry for entry in state.get("results", []) if entry["id"] == result_id), None)
    if result is None:
        raise Http404("Результат не найден в вашей сессии.")
    return FileResponse((settings.MEDIA_ROOT / result["image"]).open("rb"), as_attachment=True,
                        filename=f"pixelnaya-{result['operation']}.png", content_type="image/png")
