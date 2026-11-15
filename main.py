import os
import subprocess
import sys
import flet as ft
from levels_config import LEVELS


def main(page: ft.Page):
    page.title = "Вчимося читати з Капітошкою"
    page.bgcolor = "#E6F2FF"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.window_width = 800
    page.window_height = 600

    status_text = ft.Text("", size=16, color=ft.Colors.BLUE_900)

    def go_to_level(e, level_id):
        if not LEVELS[level_id]["is_unlocked"]:
            status_text.value = f"Замочок «{LEVELS[level_id]['letter']}» ще закритий! Спочатку пройди попередні рівні."
            page.update()
            return

        target_script = "ui_app1.py" if level_id == 1 else "level_u.py"
        if os.path.exists(target_script):
            status_text.value = f"Запускаємо рівень {level_id}: Буква {LEVELS[level_id]['letter']}..."
            page.update()
            subprocess.Popen([sys.executable, target_script])
        else:
            status_text.value = f"Перехід на рівень {level_id}: Буква {LEVELS[level_id]['letter']}"
            page.update()

    locks_row_1 = ft.Row(alignment=ft.MainAxisAlignment.CENTER, spacing=20)
    locks_row_2 = ft.Row(alignment=ft.MainAxisAlignment.CENTER, spacing=20)

    for level_id, data in LEVELS.items():
        unlocked = data["is_unlocked"]
        btn_content = ft.Column(
            controls=[
                ft.Text(
                    data["letter"],
                    size=40,
                    weight=ft.FontWeight.BOLD,
                    color=data["color"] if unlocked else ft.Colors.GREY_400,
                ),
                ft.Icon(
                    name=ft.Icons.LOCK_OPEN if unlocked else ft.Icons.LOCK,
                    color=ft.Colors.GREEN if unlocked else ft.Colors.GREY_400,
                    size=30,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        level_container = ft.Container(
            content=btn_content,
            width=100,
            height=100,
            bgcolor=ft.Colors.WHITE,
            border_radius=15,
            shadow=ft.BoxShadow(spread_radius=1, blur_radius=5, color=ft.Colors.BLACK12),
            on_click=lambda e, lid=level_id: go_to_level(e, lid),
            ink=unlocked,
        )

        if level_id <= 5:
            locks_row_1.controls.append(level_container)
        else:
            locks_row_2.controls.append(level_container)

    page.add(
        ft.Text("Обери літеру!", size=35, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700),
        ft.Container(height=30),
        locks_row_1,
        ft.Container(height=20),
        locks_row_2,
        ft.Container(height=20),
        status_text,
    )


ft.app(target=main)