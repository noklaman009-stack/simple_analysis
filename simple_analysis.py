"""SimpleAnalysis: генерация данных, вычисления и визуализация.

Скрипт выполняет полный цикл:
- генерирует Series из 1000 целых чисел в диапазоне [-10000, 10000];
- рассчитывает базовые статистики;
- строит графики (линейный и гистограмму с округлением до сотен);
- формирует DataFrame с отсортированными рядами и визуализирует их.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки.

    Возвращает:
        argparse.Namespace: набор параметров для запуска.
    """
    parser = argparse.ArgumentParser(
        description=(
            "SimpleAnalysis: генерация данных, вычисления и визуализация "
            "(все действия выполняются автоматически)."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Фиксированный seed для генератора случайных чисел.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Каталог для сохранения графиков.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не показывать графики в интерактивном окне.",
    )
    return parser.parse_args()


def generate_series(size: int = 1000, low: int = -10000, high: int = 10000,
                    seed: int | None = None) -> pd.Series:
    """Генерирует Series со случайными целыми числами.

    Args:
        size (int): количество чисел.
        low (int): минимальное значение (включительно).
        high (int): максимальное значение (включительно).
        seed (int | None): seed для воспроизводимости.

    Returns:
        pd.Series: случайный набор данных.
    """
    rng = np.random.default_rng(seed)
    values = rng.integers(low, high + 1, size=size)
    return pd.Series(values, name="values")


def count_repeated(series: pd.Series) -> int:
    """Подсчитывает количество повторяющихся значений.

    Под повторяющимися понимаются все элементы, кроме первого в каждой группе
    одинаковых значений. Например, если число встретилось 3 раза, то повторов 2.

    Args:
        series (pd.Series): исходные данные.

    Returns:
        int: количество повторов.
    """
    value_counts = series.value_counts()
    repeats_per_value = value_counts[value_counts > 1] - 1
    return int(repeats_per_value.sum())


def rounded_to_hundreds(values: Iterable[int]) -> np.ndarray:
    """Округляет значения до сотен по математическому правилу.

    Args:
        values (Iterable[int]): значения для округления.

    Returns:
        np.ndarray: округлённые значения.
    """
    return np.array([int(math.floor((x + 50) / 100) * 100) for x in values])


def print_stats(series: pd.Series) -> dict[str, float]:
    """Вычисляет и печатает базовые статистики для Series.

    Args:
        series (pd.Series): исходные данные.

    Returns:
        dict[str, float]: словарь статистических значений.
    """
    minimum = int(series.min())
    maximum = int(series.max())
    total_sum = int(series.sum())
    std_dev = float(series.std(ddof=0))
    repeats = count_repeated(series)

    print("\n=== Результаты анализа данных ===")
    print(f"Минимальное значение: {minimum}")
    print(f"Максимальное значение: {maximum}")
    print(f"Сумма чисел: {total_sum}")
    print(f"Среднеквадратическое отклонение: {std_dev:.2f}")
    print(f"Количество повторяющихся значений: {repeats}")

    return {
        "min": minimum,
        "max": maximum,
        "sum": total_sum,
        "std": std_dev,
        "repeats": repeats,
    }


def apply_gradient_background(ax: plt.Axes, colors: tuple[str, str]) -> None:
    """Добавляет мягкий градиентный фон на график.

    Args:
        ax (plt.Axes): ось, к которой добавляется фон.
        colors (tuple[str, str]): два цвета для градиента.
    """
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=plt.get_cmap(
            "coolwarm" if colors == ("#1e3c72", "#2a5298") else "viridis"
        ),
        extent=[0, 1, 0, 1],
        transform=ax.transAxes,
        zorder=0,
        alpha=0.15,
    )


def plot_line(series: pd.Series, output_path: Path, show: bool) -> None:
    """Строит линейный график исходных данных.

    Args:
        series (pd.Series): исходные данные.
        output_path (Path): путь для сохранения.
        show (bool): показывать ли окно с графиком.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    apply_gradient_background(ax, ("#1e3c72", "#2a5298"))

    x_values = np.arange(len(series))
    ax.plot(
        x_values,
        series.values,
        color="#ff6b6b",
        linewidth=1.3,
        label="Случайные значения",
    )
    ax.set_title("Линейный график набора данных", fontsize=14, weight="bold")
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_histogram(series: pd.Series, output_path: Path, show: bool) -> None:
    """Строит гистограмму для округленных значений.

    Args:
        series (pd.Series): исходные данные.
        output_path (Path): путь для сохранения.
        show (bool): показывать ли окно с графиком.
    """
    rounded = rounded_to_hundreds(series.values)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    apply_gradient_background(ax, ("#11998e", "#38ef7d"))

    counts, bins, patches = ax.hist(
        rounded,
        bins=30,
        edgecolor="#222",
        linewidth=0.6,
        alpha=0.85,
    )

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    spread = np.ptp(bin_centers) or 1
    colors = plt.cm.viridis((bin_centers - bin_centers.min()) / spread)
    for patch, color in zip(patches, colors, strict=True):
        patch.set_facecolor(color)

    ax.set_title("Гистограмма (округление до сотен)", fontsize=14, weight="bold")
    ax.set_xlabel("Округлённое значение")
    ax.set_ylabel("Частота")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def build_dataframe(series: pd.Series) -> pd.DataFrame:
    """Создаёт DataFrame с исходными и отсортированными данными.

    Args:
        series (pd.Series): исходные данные.

    Returns:
        pd.DataFrame: таблица с добавленными столбцами.
    """
    df = pd.DataFrame({"original": series})
    df["sorted_asc"] = series.sort_values(ignore_index=True)
    df["sorted_desc"] = series.sort_values(ascending=False, ignore_index=True)
    return df


def plot_sorted(df: pd.DataFrame, output_path: Path, show: bool) -> None:
    """Строит два линейных графика отсортированных рядов на одном plt.

    Args:
        df (pd.DataFrame): таблица с данными.
        output_path (Path): путь для сохранения.
        show (bool): показывать ли окно с графиком.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    apply_gradient_background(ax, ("#2b5876", "#4e4376"))

    ax.plot(
        df["sorted_asc"].values,
        color="#00b4d8",
        linewidth=1.6,
        label="По возрастанию",
    )
    ax.plot(
        df["sorted_desc"].values,
        color="#fcbf49",
        linewidth=1.6,
        label="По убыванию",
    )

    ax.set_title("Сравнение отсортированных рядов", fontsize=14, weight="bold")
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Точка входа: выполняет генерацию, анализ и визуализацию."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    series = generate_series(seed=args.seed)
    print_stats(series)

    df = build_dataframe(series)

    plot_line(series, output_dir / "line_plot.png", show=not args.no_show)
    plot_histogram(series, output_dir / "histogram.png", show=not args.no_show)
    plot_sorted(df, output_dir / "sorted_lines.png", show=not args.no_show)

    print("\nГрафики сохранены в:")
    print(f"- {output_dir / 'line_plot.png'}")
    print(f"- {output_dir / 'histogram.png'}")
    print(f"- {output_dir / 'sorted_lines.png'}")


if __name__ == "__main__":
    main()