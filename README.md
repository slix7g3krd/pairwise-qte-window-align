# pairwise-qte-window-align

## 领域

游戏、图形与多媒体

## 主题

快速反应事件窗口对齐

## 说明

逻辑时间轴窗口、输入时间戳换算与暂停冻结。

初始状态下 `gameplay/qte_window_align.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
