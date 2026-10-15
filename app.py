import sys
import os
import threading
import tkinter as tk
from controller import MainApp

# 👉 防止 PyInstaller 无控制台时报错（保留你原来的）
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    root = tk.Tk()
    app = MainApp(root)

    def on_close():
        print("正在关闭程序...")

        try:
            # ✅ 1. 通知业务层关闭（推荐）
            if hasattr(app, "on_close"):
                app.on_close()

            # ✅ 2. 等待非主线程结束（最多等1秒，避免卡死）
            for t in threading.enumerate():
                if t is not threading.main_thread():
                    print(f"等待线程结束: {t.name}")
                    t.join(timeout=1)

        except Exception as e:
            print("关闭时异常:", e)

        try:
            # ✅ 3. 销毁窗口
            root.destroy()
        except:
            pass

        # ✅ 4. 正常退出
        try:
            sys.exit(0)
        except:
            pass

        # ✅ 5. 最后兜底（防止顽固线程）
        os._exit(0)

    # ⭐ 核心：拦截窗口关闭按钮
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


if __name__ == "__main__":
    main()