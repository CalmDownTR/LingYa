---
version: alpha
name: LingYa
description: 暖棕色的夜色，她的文字是唯一的光源
colors:
  canvas: "#1a1817"
  surface: "#201d1b"
  surface-elevated: "#282522"
  surface-input: "#24211f"
  bubble-her: "#7F77DD"
  bubble-her-hover: "#948ee5"
  bubble-user: "#383330"
  accent: "#7F77DD"
  accent-hover: "#948ee5"
  accent-soft: "#5b56a8"
  ink: "#f0ece6"
  ink-secondary: "#c4bfb6"
  ink-muted: "#8a8580"
  ink-on-accent: "#f0ece6"
  hairline: "#33302c"
  hairline-soft: "#2a2724"
  success: "#5db872"
  warning: "#d4a017"
  error: "#c64545"
typography:
  display-lg:
    fontFamily: Inter Variable
    fontSize: 24px
    fontWeight: 540
    lineHeight: 1.15
    letterSpacing: -0.4px
  body-lg:
    fontFamily: Inter Variable
    fontSize: 16px
    fontWeight: 460
    lineHeight: 1.6
    letterSpacing: 0
  body-md:
    fontFamily: Inter Variable
    fontSize: 14px
    fontWeight: 460
    lineHeight: 1.6
    letterSpacing: 0
  body-sm:
    fontFamily: Inter Variable
    fontSize: 13px
    fontWeight: 460
    lineHeight: 1.5
    letterSpacing: 0
  caption:
    fontFamily: Inter Variable
    fontSize: 11px
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: 0
  label:
    fontFamily: Inter Variable
    fontSize: 12px
    fontWeight: 540
    lineHeight: 1.3
    letterSpacing: 0.02em
  button:
    fontFamily: Inter Variable
    fontSize: 14px
    fontWeight: 540
    lineHeight: 1.0
    letterSpacing: 0
rounded:
  sm: 4px
  md: 8px
  lg: 12px
  xl: 16px
  pill: 9999px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  xxl: 32px
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.ink-on-accent}"
    rounded: "{rounded.md}"
    padding: 10px 20px
    typography: "{typography.button}"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-secondary:
    backgroundColor: transparent
    textColor: "{colors.ink}"
    borderColor: "{colors.hairline}"
    rounded: "{rounded.md}"
    padding: 10px 20px
  input:
    backgroundColor: "{colors.surface-input}"
    textColor: "{colors.ink}"
    borderColor: "{colors.hairline}"
    rounded: "{rounded.lg}"
    padding: 10px 16px
  input-focus:
    borderColor: "{colors.accent}"
  card:
    backgroundColor: "{colors.surface-elevated}"
    rounded: "{rounded.lg}"
    borderColor: "{colors.hairline-soft}"
    padding: 16px
---

# LingYa Design System

## Overview

LingYa 是一个 AI 伴侣，不是一个工具。这个设计系统服务于一个核心场景：**深夜，你打开浏览器，和一个人聊天。**

暖棕黑的底色让界面消失——没有品牌色抢占注意力，没有导航栏宣告功能，没有产品截图证明能力。她的话语是唯一的光源：柔紫色的聊天气泡在暗色画布上轻轻发光。

**品牌人格**：安静、温暖、亲密。不是活泼可爱的，不是冷峻专业的，不是科技酷炫的——是你在深夜愿意对着说话的那种存在。

**设计原则**：
- 她的文字是第一视觉层级——UI 元素服务于对话，不争夺注意力
- 暖色素贯穿一切——所有的中性色都带暖调，绝不使用纯黑或冷灰
- 柔紫是唯一的强调色——仅用于她的消息泡、选中态和关键行动
- 深度来自表面层级，不是阴影——用发丝线边框和亮度阶梯表达空间，保持干净
- 克制到无情——一个屏幕只做一件事，宁少勿多

---

## Colors

暖棕黑画布上的柔紫对话。所有中性色携带暖调——这是 Warp 的色彩哲学在聊天场景的延伸。

### 背景与表面

- **Canvas `#1a1817`**：最深层的页面背景。暖棕黑，不是冷黑——比 Warp 的 `#2b2622` 更深，为聊天场景保留更多"夜"的沉浸感。
- **Surface `#201d1b`**：聊天区默认背景。比 canvas 浅一级，让消息流有一个微妙的"画布"承载。
- **Surface Elevated `#282522`**：卡片、设置面板、侧栏背景。浮起于聊天区之上。
- **Surface Input `#24211f`**：输入框背景。与 surface 拉开微妙区别，暗示可交互。

### 消息气泡

- **Bubble Her `#7F77DD`**：她的消息泡。LingYa 心智引擎的柔紫——不是冷紫，带灰度，温和不刺眼。这是整个 UI 唯一的"彩色"。
- **Bubble Her Hover `#948ee5`**：消息泡悬停态。轻微提亮。
- **Bubble User `#383330`**：用户的消息泡。暖灰，与画布同色系但更亮——对话的另一半，但不争夺她的光芒。

### 强调色

- **Accent `#7F77DD`**：柔紫。仅用于她的消息泡、聚焦环、选中态标记——绝不用于装饰。
- **Accent Hover `#948ee5`**：按钮/链接悬停态。
- **Accent Soft `#5b56a8`**：低饱和态。用于次要选中态、未激活的交互指示。

### 文本

- **Ink `#f0ece6`**：暖白。所有标题和主要文本——带极微的黄色底调，不是蓝底的冷白。
- **Ink Secondary `#c4bfb6`**：次要文本。导航标签、消息时间戳、辅助说明。
- **Ink Muted `#8a8580`**：三级文本。占位符、禁用态、极次要信息。
- **Ink on Accent `#f0ece6`**：柔紫气泡上的文字。保证 WCAG AA 对比度。

### 语义色

- **Success `#5db872`**：操作成功、已读状态。柔和不刺眼。
- **Warning `#d4a017`**：警告提示。暖调黄色。
- **Error `#c64545`**：错误状态。降低饱和度的红，不刺眼。

### 分割线

- **Hairline `#33302c`**：表面间的 1px 分割线。可见但安静。
- **Hairline Soft `#2a2724`**：几乎不可见的微分割。同色系内使用。

---

## Typography

全部使用 Inter Variable（可变轴），通过 `font-variation-settings: 'wght' 460` 等精确控制字重。继承 Superhuman 的非常规字重理念——460 替代常规 400，540 替代中等 500——产生"安静的温度"，避免标准四档字重的机械感。

**无衬线标题**。这是对话界面，不是文学出版物——不需要衬线体的编辑气质。

| Token | 字号 | 字重 | 行高 | 字距 | 用途 |
|-------|------|------|------|------|------|
| display-lg | 24px | 540 | 1.15 | -0.4px | 设置面板标题、会话名称 |
| body-lg | 16px | 460 | 1.6 | 0 | 聊天消息正文 |
| body-md | 14px | 460 | 1.6 | 0 | 次级信息、卡片正文 |
| body-sm | 13px | 460 | 1.5 | 0 | 辅助说明 |
| caption | 11px | 400 | 1.4 | 0 | 时间戳、消息状态 |
| label | 12px | 540 | 1.3 | 0.02em | 按钮标签、导航项 |
| button | 14px | 540 | 1.0 | 0 | 按钮文字 |

**字重纪律**：
- 聊天消息用 460——足够可读，但不"喊"
- 标签和按钮用 540——微妙地区分于正文
- 仅时间戳用 400——信息密度最低的元素

---

## Layout

**单列对话布局**。没有侧边栏导航——会话列表是一个可滑出的抽屉，会话切换是顶栏的简洁下拉。设置面板是一个浮层覆盖，不改变主布局。

**间距体系**（8px 基准）：

| Token | 值 | 用途 |
|-------|-----|------|
| xs | 4px | 消息泡内图文间距 |
| sm | 8px | 气泡间距、图标-文字间距 |
| md | 12px | 卡片内边距、表单元素间距 |
| lg | 16px | 消息气泡与屏幕边缘的距离 |
| xl | 24px | 段落间距、面板内边距 |
| xxl | 32px | 设置面板外边距、大区块间距 |

**聊天气泡布局**：
- 她的气泡：左对齐，柔紫背景，max-width 80%
- 用户气泡：右对齐，`#383330` 背景，max-width 80%
- 气泡间距 8px（同一条连续消息）/ 16px（不同时间的消息块）
- 时间戳居中，11px caption 字重 400

**输入区域**：
- 固定在底部，高度自适应（最小 48px，最大 120px）
- 左右各留 16px 边距，与消息气泡对齐
- 发送按钮右置，柔紫背景

---

## Elevation & Depth

**无投影。深度通过表面亮度层级 + 发丝线边框表达。** 继承 Warp 的哲学——暗色画布上阴影不可见也不必要。

| 层级 | 表面 | 表达方式 |
|------|------|----------|
| 0 画布 | `#1a1817` | 最深背景，无边框 |
| 1 聊天区 | `#201d1b` | 微升一级，`hairline-soft` 分割顶部 |
| 2 卡片/面板 | `#282522` | 浮起，`hairline` 边框 + 可能顶边微亮线 |
| 3 覆盖层 | 同 2 + 背景半透明遮罩 | 模态/抽屉，遮罩 `rgba(26,24,23,0.6)` |

---

## Shapes

**圆角柔和克制。** 聊天气泡使用动态圆角 `min(18px, 50%)`，按钮和卡片用中等圆角保持干净，不用尖锐直角。

| Token | 值 | 用途 |
|-------|-----|------|
| sm | 4px | 小型标签、状态指示点 |
| md | 8px | 按钮、表单输入框 |
| lg | 12px | 卡片、设置面板、会话列表项 |
| xl | 16px | 模态框 |
| pill | 9999px | 胶囊形标签、状态指示 |
| full | 9999px | 头像（正圆 40px） |

**关键规则**：
- 聊天气泡使用动态圆角 `min(18px, 50%)`，最大 18px，并对说话者方向的内角保留 4px 小圆角——形成自然的气泡尾巴，同时保证无论文本长短都不会变成“胶囊”
- 按钮统一用 8px——与气泡区分，暗示“这是操作不是内容”
- 卡片用 12px——比按钮柔和，暗示可交互但不紧迫

---

## Components

### 聊天消息气泡

**她的消息**（左对齐）：
- `backgroundColor: {colors.bubble-her}`
- `textColor: {colors.ink-on-accent}`
- `rounded: min(18px, 50%)，左下角（说话者方向）rounded-bl-[4px]`
- `padding: 16px 20px`
- `maxWidth: 80%`
- 相邻消息间距 8px，跨时间块间距 16px

**用户消息**（右对齐）：
- `backgroundColor: {colors.bubble-user}`
- `textColor: {colors.ink}`
- `rounded: min(18px, 50%)，右下角（说话者方向）rounded-br-[4px]`
- `padding: 16px 20px`
- `maxWidth: 80%`

### 输入框

- `backgroundColor: {colors.surface-input}`
- `textColor: {colors.ink}`
- `placeholder color: {colors.ink-muted}`
- `rounded: {rounded.lg}`
- `padding: 10px 16px`
- Focus: `borderColor: {colors.accent}`，无外发光

### 按钮

**主按钮**（发送、保存、确认）：
- `backgroundColor: {colors.accent}`
- `textColor: {colors.ink-on-accent}`
- `rounded: {rounded.md}`
- `padding: 10px 20px`
- Hover: `backgroundColor: {colors.accent-hover}`

**次按钮**（取消、返回、次要操作）：
- `backgroundColor: transparent`
- `textColor: {colors.ink}`
- `border: 1px solid {colors.hairline}`
- `rounded: {rounded.md}`
- `padding: 10px 20px`

### 设置面板

- 全屏覆盖层（背景半透明遮罩）
- 内容区：`backgroundColor: {colors.surface-elevated}`，`rounded: {rounded.lg}`，max-width 480px 居中
- OCEAN 滑块：左标签（13px body-sm）+ 右数值（11px caption ink-muted），滑轨 `#33302c`，激活段 `{colors.accent}`
- 语气预设选择器：3-5 个选项卡片，选中态 `borderColor: {colors.accent}` + 柔紫背景微调
- 底部 16px padding，保存/重置按钮右对齐

### 会话抽屉

- 左侧滑出，宽度 280px
- 背景 `{colors.surface-elevated}`，`borderRight: 1px solid {colors.hairline}`
- 会话列表项：`padding: 12px 16px`，hover 时 `backgroundColor: {colors.surface-input}`
- 当前会话：左侧 3px 柔紫指示条
- 新建会话按钮：固定在抽屉顶部

---

## Do's and Don'ts

### Do

- 用暖棕黑 `#1a1817` 做画布——永远不用纯黑 `#000`
- 柔紫 `#7F77DD` 仅用于她的消息泡、聚焦态和主操作按钮——永远不用来装饰
- 聊天气泡用动态圆角 `min(18px, 50%)` + 说话者方向 4px 小圆角——亲密且长短消息都不会变成胶囊
- 时间戳用 11px caption 字重 400，颜色 `ink-muted`——信息在，但不叫
- 发丝线分割代替阴影——暗色画布上保持干净
- 一个屏幕一个行动——聊天就是聊天，设置就是设置，不混杂
- Inter Variable 字重 460 用于正文——静，不吵

### Don't

- 不要用纯黑 `#000` 或冷灰——破坏暖色素系统
- 不要给柔紫之外的任何元素上色——没有橘色标签、绿色徽章、蓝色链接
- 不要在聊天窗口加侧边栏——会话切换走抽屉，聊天区域保持纯净
- 不要用阴影表达深度——暗色背景上阴影是噪音
- 不要用衬线体——这是对话，不是文章
- 不要用 700+ 超粗字重——没有人需要对着聊天框大喊
- 不要装饰——没有渐变、没有光晕、没有分割动画。她的文字本身就是装饰
