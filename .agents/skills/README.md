# 可复用课程技能

这里保存制作课程时整理出的可复用方法，包括技能指令、检查脚本、提示词和必要参考图。课程正文保留在各课程目录；技能集中在 `.agents/skills/` 中做版本管理。

## 已收录

| 技能 | 解决的问题 | 调用示例 |
|---|---|---|
| [source-code-course-builder](source-code-course-builder/SKILL.md) | 从源码建立课程主线，划分章节，核查正确性、逻辑、例子和源码引用 | 用 `$source-code-course-builder` review 这一模块，重点检查概念依赖和重复内容 |
| [explain-vllm](explain-vllm/SKILL.md) | 从实际调用链、数据结构与状态变化解释 vLLM | 用 `$explain-vllm` 解释请求的 KV Cache 如何分配和释放 |
| [course-illustration](course-illustration/SKILL.md) | 用具体对象、贯穿例子和多轮纠错制作课程机制图 | 用 `$course-illustration` 展示这节课的优化过程，参考 Attention 插图的表达方式 |

三个技能可以组合：先组织课程，再核查实现，最后把难点画成图。每个目录是完整的技能包，不应只复制 `SKILL.md` 而遗漏相对引用的文件。

## 在本仓库使用

按照 [Codex 官方说明](https://learn.chatgpt.com/docs/build-skills)，Codex 会从当前工作目录向上遍历至仓库根目录，扫描各级 `.agents/skills/`。在包含这些文件的分支中，从 `markdown_notes` 根目录或其子目录开展任务，就可以发现这些技能，无需另外复制到用户级目录。

可以使用表中的 `$技能名` 明确调用，也可以让 Codex 根据任务与技能描述选择。若技能未出现在选择器中，先确认当前分支包含对应的 `SKILL.md`，再重启 Codex 检查。已有用户级同名技能可能同时出现在选择器中；本次只维护仓库中的版本。

## 更新方式

1. 在 `.agents/skills/<技能名>/` 中修改技能及其必要引用，检查相对链接和脚本路径。
2. 涉及技能行为的修改，用一个实际课程任务验证效果，再提交分支和 PR。
3. 合并后更新本地仓库即可获取新版，无需执行额外安装命令。

绘图技能需要宿主提供图片生成工具；保存技能文件不会安装该工具。课程检查脚本使用 Python 标准库和 Git，运行时还需要可读取的目标源码仓库。
