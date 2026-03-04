"""学习计划 API — 目标管理、学习记录、进度追踪"""
from fastapi import APIRouter, Request, HTTPException
from sqlalchemy import select, func
from datetime import datetime, timedelta
from loguru import logger

from app.models.schemas import LearningGoal, StudySession, ProgressReport
from app.models.db_models import LearningGoalDB, StudySessionDB
from app.core.database import async_session

router = APIRouter()


@router.post("/goals")
async def create_goal(goal: LearningGoal):
    """创建学习目标"""
    async with async_session() as session:
        db_goal = LearningGoalDB(
            topic=goal.topic,
            target=goal.target,
            deadline=goal.deadline,
            daily_minutes=goal.daily_minutes,
        )
        session.add(db_goal)
        await session.commit()
        
        return {
            "id": db_goal.id,
            "topic": db_goal.topic,
            "target": db_goal.target,
            "daily_minutes": db_goal.daily_minutes,
            "status": db_goal.status,
        }


@router.get("/goals")
async def list_goals(status: str = "active"):
    """列出学习目标"""
    async with async_session() as session:
        query = select(LearningGoalDB)
        if status:
            query = query.where(LearningGoalDB.status == status)
        result = await session.execute(query.order_by(LearningGoalDB.created_at.desc()))
        goals = result.scalars().all()
        return [
            {
                "id": g.id,
                "topic": g.topic,
                "target": g.target,
                "deadline": g.deadline.isoformat() if g.deadline else None,
                "daily_minutes": g.daily_minutes,
                "status": g.status,
                "created_at": g.created_at.isoformat() if g.created_at else None,
            }
            for g in goals
        ]


@router.post("/sessions")
async def log_study_session(session_data: StudySession):
    """记录学习会话"""
    async with async_session() as session:
        # 验证目标存在
        result = await session.execute(
            select(LearningGoalDB).where(LearningGoalDB.id == session_data.goal_id)
        )
        goal = result.scalar_one_or_none()
        if not goal:
            raise HTTPException(status_code=404, detail="学习目标不存在")

        db_session = StudySessionDB(
            goal_id=session_data.goal_id,
            duration_minutes=session_data.duration_minutes,
            notes=session_data.notes,
            difficulty=session_data.difficulty,
        )
        session.add(db_session)
        await session.commit()

        return {
            "id": db_session.id,
            "goal_id": db_session.goal_id,
            "duration_minutes": db_session.duration_minutes,
            "message": f"已记录 {db_session.duration_minutes} 分钟学习",
        }


@router.get("/progress/{goal_id}")
async def get_progress(goal_id: str, request: Request):
    """获取学习进度报告（含 AI 教练反馈）"""
    async with async_session() as session:
        # 获取目标
        result = await session.execute(
            select(LearningGoalDB).where(LearningGoalDB.id == goal_id)
        )
        goal = result.scalar_one_or_none()
        if not goal:
            raise HTTPException(status_code=404, detail="学习目标不存在")

        # 获取所有学习记录
        result = await session.execute(
            select(StudySessionDB)
            .where(StudySessionDB.goal_id == goal_id)
            .order_by(StudySessionDB.created_at.desc())
        )
        sessions = result.scalars().all()

        # 计算统计数据
        total_minutes = sum(s.duration_minutes for s in sessions)
        total_hours = round(total_minutes / 60, 1)

        # 计算连续学习天数
        study_dates = set()
        for s in sessions:
            if s.created_at:
                study_dates.add(s.created_at.date())
        
        streak = 0
        today = datetime.now().date()
        check_date = today
        while check_date in study_dates:
            streak += 1
            check_date -= timedelta(days=1)

        # 计算完成度（基于截止日期和每日目标）
        if goal.deadline and goal.created_at:
            total_days = (goal.deadline - goal.created_at).days or 1
            expected_minutes = total_days * goal.daily_minutes
            completion_pct = min(round(total_minutes / expected_minutes * 100, 1), 100)
        else:
            completion_pct = 0

        # AI 教练反馈
        avg_difficulty = (
            sum(s.difficulty for s in sessions) / len(sessions)
            if sessions else 3
        )
        
        rag_engine = request.app.state.rag_engine
        coach_prompt = f"""作为学习教练，请基于以下数据给出简短的反馈和建议：
        - 学习主题：{goal.topic}
        - 学习目标：{goal.target}
        - 已学习：{total_hours} 小时
        - 连续学习：{streak} 天
        - 完成度：{completion_pct}%
        - 平均难度感受：{avg_difficulty}/5
        - 最近笔记：{sessions[0].notes if sessions else '无'}
        
        请给出：1) 一句鼓励的话 2) 2-3 条具体建议"""

        try:
            from llama_index.llms.openai import OpenAI
            from app.core.config import get_settings
            settings = get_settings()
            llm = OpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY)
            coach_response = llm.complete(coach_prompt)
            coach_feedback = str(coach_response)
        except Exception as e:
            logger.warning(f"AI 反馈生成失败: {e}")
            coach_feedback = f"你已经学习了 {total_hours} 小时，连续 {streak} 天，继续保持！"

        return {
            "goal_id": goal_id,
            "topic": goal.topic,
            "total_hours": total_hours,
            "streak_days": streak,
            "completion_pct": completion_pct,
            "session_count": len(sessions),
            "avg_difficulty": round(avg_difficulty, 1),
            "coach_feedback": coach_feedback,
        }


@router.get("/sessions/{goal_id}")
async def list_sessions(goal_id: str, limit: int = 20):
    """列出学习记录"""
    async with async_session() as session:
        result = await session.execute(
            select(StudySessionDB)
            .where(StudySessionDB.goal_id == goal_id)
            .order_by(StudySessionDB.created_at.desc())
            .limit(limit)
        )
        sessions = result.scalars().all()
        return [
            {
                "id": s.id,
                "duration_minutes": s.duration_minutes,
                "notes": s.notes,
                "difficulty": s.difficulty,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in sessions
        ]
