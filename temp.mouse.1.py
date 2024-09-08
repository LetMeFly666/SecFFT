'''
Author: LetMeFly
Date: 2024-09-08 02:53:30
LastEditors: LetMeFly
LastEditTime: 2024-09-08 02:53:41
'''
from pynput import mouse
import pyautogui

# 定义鼠标单击事件的回调函数
def on_click(x, y, button, pressed):
    if button == mouse.Button.left and pressed:  # 只处理左键按下事件
        # 获取当前鼠标位置
        current_position = pyautogui.position()
        print(f"鼠标当前位置: {current_position}")
        return False  # 返回False以停止监听器

# 监听鼠标事件
with mouse.Listener(on_click=on_click) as listener:
    listener.join()
