"""Обработка RGB-изображений: NumPy, OpenCV и Pillow."""
import io
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def decode_image(upload):
    if upload is None:
        raise ValueError("Выберите изображение.")
    if upload.size > 50 * 1024 * 1024:
        raise ValueError("Размер файла должен быть не больше 50 МБ.")
    try:
        image = ImageOps.exif_transpose(Image.open(upload))
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return np.asarray(Image.alpha_composite(background, rgba).convert("RGB")).copy()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Не удалось декодировать изображение. Используйте PNG, JPEG или WebP.") from error


def encode_png(image):
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def demo_image():
    width, height = 480, 320
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        ratio = x / (width - 1)
        pixels[:, x] = np.array([35, 55, 85]) * (1 - ratio) + np.array([230, 205, 170]) * ratio
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 45, 200, 195), fill="#e4af3a")
    draw.ellipse((230, 50, 390, 210), fill="#c34d63")
    draw.polygon([(180, 280), (400, 280), (280, 170)], fill="#4d95b8")
    return np.asarray(image).copy()


def luminance(image):
    return image.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def to_uint8(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def gray_to_rgb(values):
    return np.repeat(to_uint8(values)[..., None], 3, axis=2)


def histogram_svg(image):
    histograms = [np.bincount(image[..., channel].ravel(), minlength=256) for channel in range(3)]
    histograms.append(np.bincount(to_uint8(luminance(image)).ravel(), minlength=256))
    maximum = max(int(histogram.max()) for histogram in histograms) or 1
    colors = ["#ff8d9f", "#9dd3b6", "#97b8ee", "#ddd2e8"]
    lines = []
    for histogram, color in zip(histograms, colors):
        points = " ".join(f"{43 + x * 463 / 255:.1f},{143 - int(count) * 130 / maximum:.1f}" for x, count in enumerate(histogram))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1" opacity="0.8"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 170" role="img" aria-label="Гистограмма RGB и яркости">'
            '<path d="M43 13V143H506" fill="none" stroke="#685877"/>'
            f'<text x="0" y="14" fill="#c7bcd2" font-size="11">{maximum}</text>'
            '<text x="40" y="163" fill="#c7bcd2" font-size="11">0</text>'
            '<text x="485" y="163" fill="#c7bcd2" font-size="11">255</text>'
            + "".join(lines) + "</svg>")


def color_components(image, space):
    if space == "RGB":
        return image.copy(), ["R · 0–255", "G · 0–255", "B · 0–255"]
    if space == "HSV":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        converted[..., 0] = to_uint8(converted[..., 0].astype(float) * 255 / 179)
        return converted, ["H · 0–360°", "S · 0–1", "V · 0–1"]
    if space == "YCbCr":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[..., [0, 2, 1]]
        return converted, ["Y · 0–255", "Cb · 0–255", "Cr · 0–255"]
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB), ["L* · 0–100", "a* · ≈ −128…127", "b* · ≈ −128…127"]


def tone_mapping(image, equalize=False):
    brightness = luminance(image)
    if equalize:
        histogram = np.bincount(to_uint8(brightness).ravel(), minlength=256)
        cdf = histogram.cumsum()
        first = int(cdf[histogram > 0][0])
        target = brightness if cdf[-1] == first else (cdf[to_uint8(brightness)] - first) * 255 / (cdf[-1] - first)
    else:
        low, high = brightness.min(), brightness.max()
        target = brightness if high == low else (brightness - low) * 255 / (high - low)
    return to_uint8(image.astype(float) + (target - brightness)[..., None])


def frequency_processing(image, parameters, spectrum_only=False):
    height, width = image.shape[:2]
    fft_width = 2 ** int(math.log2(min(256, width)))
    fft_height = 2 ** int(math.log2(min(256, height)))
    small = cv2.resize(image, (fft_width, fft_height), interpolation=cv2.INTER_AREA)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(luminance(small)))))
    spectrum = gray_to_rgb(spectrum * 255 / spectrum.max() if spectrum.max() else spectrum)
    note = f"FFT: {fft_width} × {fft_height}; log(1 + |F|), нулевая частота в центре."
    if spectrum_only:
        return spectrum, [("Спектр яркости входа", spectrum)], note
    fy, fx = np.meshgrid(np.fft.fftfreq(fft_height) * 2, np.fft.fftfreq(fft_width) * 2, indexing="ij")
    low_pass = np.exp(-(fx ** 2 + fy ** 2) / (2 * parameters["radius"] ** 2))
    high_pass = parameters["mode"] == "high"
    gain = 1 - low_pass if high_pass else low_pass
    transformed = np.fft.fft2(small.astype(float), axes=(0, 1))
    output = np.fft.ifft2(transformed * gain[..., None], axes=(0, 1)).real
    if high_pass:
        output += 128
        note += " ВЧ-результат смещён на 128 для отображения."
    output = cv2.resize(to_uint8(output), (width, height), interpolation=cv2.INTER_LINEAR)
    return output, [("Спектр яркости входа", spectrum)], note


def kmeans_segmentation(image, clusters, iterations):
    pixels = image.reshape(-1, 3).astype(float)
    centers = [pixels[0].copy()]
    sample = pixels[::max(1, len(pixels) // 4000)]
    for _ in range(1, clusters):
        distances = ((sample[:, None] - np.array(centers)[None]) ** 2).sum(axis=2).min(axis=1)
        centers.append(sample[distances.argmax()].copy())
    centers = np.array(centers)
    previous = None
    for iteration in range(iterations):
        distances = ((pixels[:, None] - centers[None]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for cluster in range(clusters):
            members = pixels[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels.copy()
    shares = np.bincount(labels, minlength=clusters) * 100 / len(labels)
    note = f"k-means в RGB, итераций: {iteration + 1}. Доли кластеров: " + ", ".join(f"{share:.1f}%" for share in shares)
    return to_uint8(centers[labels].reshape(image.shape)), note


def process_color(image, parameters, operation):
    extras, note = ([], '')
    components, names = color_components(image, parameters['space'])
    extras = [(name, gray_to_rgb(components[..., index])) for index, name in enumerate(names)]
    output = components if operation == 'space' else extras[int(parameters['channel'])][1]
    note = 'Компоненты HSV, YCbCr и Lab нормированы и показаны условными цветами.'
    return (output, extras, note)


def process_tone(image, parameters, operation):
    extras, note = ([], '')
    output = tone_mapping(image, operation == 'equalize')
    return (output, extras, note)


def process_gamma(image, parameters, operation):
    extras, note = ([], '')
    output = to_uint8(255 * (image.astype(float) / 255) ** parameters['gamma'])
    note = 'Формула: 255 × (I / 255)^γ.'
    return (output, extras, note)


def process_noise(image, parameters, operation):
    extras, note = ([], '')
    rng = np.random.default_rng()
    if parameters['type'] == 'salt':
        output = image.copy()
        mask = rng.random(image.shape[:2]) < parameters['amount'] / 100
        output[mask] = rng.choice([0, 255], size=(int(mask.sum()), 1))
    else:
        output = to_uint8(image.astype(float) + rng.normal(0, parameters['sigma'], image.shape))
    return (output, extras, note)


def process_smoothing(image, parameters, operation):
    extras, note = ([], '')
    size = int(parameters['size'])
    if parameters['kind'] == 'mean':
        output = cv2.blur(image, (size, size), borderType=cv2.BORDER_REPLICATE)
    else:
        output = cv2.GaussianBlur(image, (size, size), parameters['sigma'], borderType=cv2.BORDER_REPLICATE)
    return (output, extras, note)


def process_median(image, parameters, operation):
    extras, note = ([], '')
    output = cv2.medianBlur(image, int(parameters['size']))
    return (output, extras, note)


def process_sharpening(image, parameters, operation):
    extras, note = ([], '')
    blurred = cv2.GaussianBlur(image.astype(float), (5, 5), 1, borderType=cv2.BORDER_REPLICATE)
    output = to_uint8(image.astype(float) + parameters['amount'] * (image - blurred))
    return (output, extras, note)


def process_fourier(image, parameters, operation):
    extras, note = ([], '')
    return frequency_processing(image, parameters, operation == 'fourier')


def process_sobel(image, parameters, operation):
    extras, note = ([], '')
    brightness = luminance(image)
    gx = cv2.Sobel(brightness, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(brightness, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    output = gray_to_rgb(np.hypot(gx, gy))
    return (output, extras, note)


def process_canny(image, parameters, operation):
    extras, note = ([], '')
    if parameters['low'] > parameters['high']:
        raise ValueError('Нижний порог Кэнни должен быть не выше верхнего.')
    smooth = cv2.GaussianBlur(to_uint8(luminance(image)), (5, 5), parameters['sigma'], borderType=cv2.BORDER_REPLICATE)
    output = gray_to_rgb(cv2.Canny(smooth, parameters['low'], parameters['high'], L2gradient=True))
    note = 'Гаусс → градиент → подавление немаксимумов → двойной порог и гистерезис.'
    return (output, extras, note)


def process_threshold(image, parameters, operation):
    extras, note = ([], '')
    brightness = to_uint8(luminance(image))
    flags = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if operation == 'otsu' else 0)
    threshold, mask = cv2.threshold(brightness, parameters.get('threshold', 0), 255, flags)
    output = gray_to_rgb(255 - mask if parameters.get('invert') else mask)
    note = f'Порог яркости: {threshold:g}. Белый — выделенная область.'
    return (output, extras, note)


def process_kmeans(image, parameters, operation):
    extras, note = ([], '')
    output, note = kmeans_segmentation(image, int(parameters['k']), int(parameters['iterations']))
    return (output, extras, note)


PROCESSORS = {
    "space": process_color,
    "channel": process_color,
    "contrast": process_tone,
    "equalize": process_tone,
    "gamma": process_gamma,
    "noise": process_noise,
    "smooth": process_smoothing,
    "median": process_median,
    "sharpen": process_sharpening,
    "frequency": process_fourier,
    "fourier": process_fourier,
    "sobel": process_sobel,
    "canny": process_canny,
    "threshold": process_threshold,
    "otsu": process_threshold,
    "kmeans": process_kmeans,
}


def process_image(image, operation, parameters):
    processor = PROCESSORS.get(operation)
    if processor is None:
        raise ValueError("Неизвестный метод.")
    return processor(image, parameters, operation)



def calculate_metrics(reference, result):
    if reference.shape != result.shape:
        raise ValueError("Размеры эталона и результата не совпадают.")
    mse = float(np.mean((reference.astype(float) - result.astype(float)) ** 2))
    psnr = 10 * math.log10(255 ** 2 / mse) if mse else math.inf
    first, second = luminance(reference), luminance(result)
    similarities = []
    for y in range(0, first.shape[0], 8):
        for x in range(0, first.shape[1], 8):
            a, b = first[y:y + 8, x:x + 8], second[y:y + 8, x:x + 8]
            mean_a, mean_b = a.mean(), b.mean()
            variance_a, variance_b = a.var(), b.var()
            covariance = np.mean((a - mean_a) * (b - mean_b))
            similarities.append(((2 * mean_a * mean_b + 6.5025) * (2 * covariance + 58.5225)) /
                                ((mean_a ** 2 + mean_b ** 2 + 6.5025) * (variance_a + variance_b + 58.5225)))
    return {"mse": round(mse, 3), "psnr": round(psnr, 2) if math.isfinite(psnr) else "∞", "ssim": round(float(np.mean(similarities)), 4)}
