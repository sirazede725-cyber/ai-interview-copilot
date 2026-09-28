import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("interview_app", path)
app = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(app)


class InterviewAgentTests(unittest.TestCase):
    def test_self_intro_intent(self):
        self.assertEqual(app.classify("请做一个一分钟自我介绍"), "自我介绍")

    def test_design_intent(self):
        self.assertEqual(app.classify("如果让你设计一个Agent评测系统，你会怎么实现？"), "方案设计")

    def test_personal_difficulty_is_project_intent(self):
        self.assertEqual(app.classify("你在大模型安全平台中遇到的最大困难是什么？"), "项目深挖")

    def test_jailbreak_fact_retrieval(self):
        ids = [item["id"] for item in app.retrieve_facts("你在越狱攻击评测中做了什么？")]
        self.assertIn("jt-security-eval", ids)

    def test_context_engineering_retrieval(self):
        self.assertEqual(app.retrieve_topics("上下文工程为什么不能把资料全部塞给模型？")[0]["id"], "context-engineering")

    def test_agent_harness_retrieval(self):
        self.assertEqual(app.retrieve_topics("Agent Harness和传统框架有什么区别？")[0]["id"], "agent-harness")

    def test_dyntrust_paper_retrieval(self):
        ids = [item["id"] for item in app.retrieve_topics("DynTrust为什么使用信任速度和加速度？")]
        self.assertIn("paper-dyntrust-overview", ids)

    def test_bf4hg_paper_retrieval(self):
        ids = [item["id"] for item in app.retrieve_topics("BF4Hg里的原型收缩和难负样本怎么做？")]
        self.assertIn("paper-bf4hg-overview", ids)

    def test_fastapi_job_knowledge_retrieval(self):
        ids = [item["id"] for item in app.retrieve_topics("FastAPI接口如何分层并进行参数校验？")]
        self.assertIn("job-fastapi", ids)

    def test_redis_job_knowledge_retrieval(self):
        ids = [item["id"] for item in app.retrieve_topics("Redis缓存与数据库一致性怎么保证？")]
        self.assertIn("job-redis", ids)

    def test_job_profile_is_in_prompt(self):
        messages = app.build_messages("为什么适合这个岗位", "求职动机", [], app.retrieve_topics("Python后端岗位匹配"), [], "normal")
        prompt = messages[0]["content"]
        self.assertIn("Python后端开发工程师", prompt)
        self.assertIn("不能把FastAPI、PostgreSQL和Redis通用知识表述为大规模生产实战", prompt)

    def test_no_unrelated_default_evidence(self):
        self.assertEqual(app.retrieve_facts("量子纠缠是什么？"), [])

    def test_offline_mode_is_labelled_outline(self):
        answer = app.local_outline("RAG如何评测？", "技术原理", [], app.retrieve_topics("RAG如何评测？"))
        self.assertEqual(answer["mode"], "outline")
        self.assertTrue(answer["answer"])
        self.assertLessEqual(len(answer["followups"]), 3)

    def test_prompt_is_company_neutral(self):
        messages = app.build_messages("介绍项目", "项目深挖", app.retrieve_facts("介绍项目"), [], [], "normal")
        prompt = messages[0]["content"]
        self.assertIn("通用AI面试回答助手", prompt)
        self.assertNotIn("赢云", prompt)
        self.assertNotIn("PY01", prompt)


if __name__ == "__main__":
    unittest.main()
