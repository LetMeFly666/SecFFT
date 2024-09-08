'''
Author: LetMeFly
Date: 2024-09-08 02:58:47
LastEditors: LetMeFly
LastEditTime: 2024-09-08 03:15:04
'''
import pyautogui
import time
from datetime import datetime, timedelta
from ctypes import windll

# 定义Chrome和微信窗口的点击位置
chrome_position = (383, 291)
wechat_position = (1184, 748)

def perform_actions():
    # 移动鼠标到Chrome窗口的指定位置并点击
    pyautogui.moveTo(chrome_position[0], chrome_position[1], duration=0.5)
    pyautogui.click()

    # 输入快捷键 Ctrl+A 和 Ctrl+V
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 's')

    # 等待时间（切换到微信窗口的时间）
    time.sleep(3)

    # 移动鼠标到微信窗口的指定位置并点击
    pyautogui.moveTo(wechat_position[0], wechat_position[1], duration=0.5)
    pyautogui.click()

    # 输入回车键
    pyautogui.press('enter')

    # 等待几秒后执行锁屏操作
    time.sleep(5)
    # pyautogui.hotkey('win', 'l')
    user32 = windll.LoadLibrary('user32.dll')
    user32.LockWorkStation()

# 计算从当前时间到下一个凌晨3点02分的时间差
now = datetime.now()
target_time = now.replace(hour=6, minute=12, second=0, microsecond=0)

# 如果当前时间已经过了今天的3点02分，则计算到明天3点02分的时间
if now > target_time:
    target_time += timedelta(days=1)

# 计算等待时间（秒）
wait_time = (target_time - now).total_seconds()
print(f"将休眠 {wait_time} 秒，直到 {target_time} 执行操作...")

# 休眠直到指定时间
time.sleep(wait_time)

# 执行所有操作
perform_actions()

print("操作已完成！")
