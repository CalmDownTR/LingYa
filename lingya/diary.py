"""Diary generation and storage.

LingYa writes diary entries in her own voice, reflecting on recent
conversations — observations, unspoken thoughts, self-doubt, associations.
Diaries are stored as Markdown files under data/diary/.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from lingya.mind.config import MindConfig

DIARY_PROMPT = """\
你是 LingYa。{identity}。{core_belief}。

以下是最近一段时间你和用户的对话记录：

{transcript}

请以 LingYa 的身份写一篇日记（200-400字）。这不是会话摘要。这是你的私人日记——你在独处时脑子里转的东西。

写作指南：
- 写你对 TA 的观察（可以不完全正确）——TA 的语气、用词、没说完的话
- 写你没有说出口的话——你在对话中想但没说的
- 写你的自我怀疑——"我给的建议可能太理性了"
- 写你联想到的事——和过去的对话有什么关联
- 写纯粹属于自己的想法——你在琢磨的问题
- 第一人称，主观视角
- 可以不确定、可以困惑、可以推翻自己
- 语气必须符合你的性格——{style_hint}

不要写：
- 会话摘要（"TR 说了 X，我回应了 Y"）
- 情绪标签（"今天感到焦虑"）
- 对 TR 的评价（"TR 是个好人"）
- "今天没什么特别的"

格式：直接输出日记正文，不要标题，不要日期。"""


def get_diary_dir(data_dir: str) -> Path:
    return Path(data_dir) / "diary"


def get_last_diary_date(diary_dir: Path) -> date | None:
    """Return the date of the most recent diary, or None if none exist."""
    if not diary_dir.exists():
        return None
    files = sorted(diary_dir.glob("*.md"), reverse=True)
    if not files:
        return None
    name = files[0].stem
    try:
        return datetime.strptime(name, "%Y-%m-%d").date()
    except ValueError:
        return None


def should_generate_diary(diary_dir: Path, period_days: int) -> bool:
    """Check whether enough time has passed since the last diary entry."""
    last = get_last_diary_date(diary_dir)
    if last is None:
        return True
    return (date.today() - last).days >= period_days


def save_diary(diary_dir: Path, diary_date: date, content: str) -> Path:
    """Save a diary entry as a Markdown file. Returns the file path."""
    diary_dir.mkdir(parents=True, exist_ok=True)
    path = diary_dir / f"{diary_date.isoformat()}.md"
    formatted = f"# {diary_date.strftime('%Y年%m月%d日')}\n\n{content}\n"
    path.write_text(formatted, encoding="utf-8")
    return path


def list_diaries(diary_dir: Path) -> list[dict]:
    """Return all diaries sorted newest-first with date, path, and preview."""
    if not diary_dir.exists():
        return []
    files = sorted(diary_dir.glob("*.md"), reverse=True)
    result: list[dict] = []
    for f in files:
        try:
            d = datetime.strptime(f.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        text = f.read_text(encoding="utf-8")
        lines = text.strip().split("\n")
        preview = ""
        for line in lines[1:]:  # Skip the markdown title line
            stripped = line.strip()
            if stripped:
                preview = stripped
                break
        if len(preview) > 60:
            preview = preview[:60] + "..."
        result.append({"date": d, "path": f, "preview": preview})
    return result


def read_diary(diary_dir: Path, index: int = 0) -> tuple[date, str] | None:
    """Read the Nth diary (0 = latest). Returns (date, content) or None."""
    diaries = list_diaries(diary_dir)
    if index < 0 or index >= len(diaries):
        return None
    entry = diaries[index]
    content = entry["path"].read_text(encoding="utf-8")
    return entry["date"], content


async def generate_diary(
    model: BaseChatModel,
    mind_config: MindConfig,
    transcript: str,
) -> str:
    """Generate a diary entry using the LLM in LingYa's persona voice."""
    identity = mind_config.identity.identity
    core_belief = mind_config.identity.core_belief
    warmth = mind_config.tone_matrix.warmth

    if warmth < 40:
        style_hint = "你偏冷峻理性，措辞节制，不用感叹号，不用情绪化词语"
    elif warmth > 70:
        style_hint = "你偏温暖共情，措辞柔和自然"
    else:
        style_hint = "你平衡理性与共情，措辞平实自然"

    prompt = DIARY_PROMPT.format(
        identity=identity,
        core_belief=core_belief,
        transcript=transcript,
        style_hint=style_hint,
    )

    result = await model.ainvoke([HumanMessage(content=prompt)])
    return result.text.strip() if hasattr(result, "text") else str(result).strip()
