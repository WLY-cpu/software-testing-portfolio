"""
路径工具模块 —— 统一管理打包前后资源路径与模块导入。
所有关卡入口文件都应从此模块获取路径，避免重复代码。
"""
import sys
import os
def get_base_dir():
    """获取项目根目录（打包后为 sys._MEIPASS，开发时为 __file__ 所在目录）。"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))
def resource_path(relative_path):
    """获取资源文件的绝对路径，兼容打包后的临时目录。"""
    return os.path.join(get_base_dir(), relative_path)
def setup_module_paths():
    """将 1~5 关卡目录加入 sys.path，方便各模块互相导入。"""
    base = get_base_dir()
    for folder in ('1', '2', '3', '4', '5'):
        folder_path = os.path.join(base, folder)
        if folder_path not in sys.path:
            sys.path.append(folder_path)
