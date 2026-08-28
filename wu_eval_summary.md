# WenetSpeech-Wu-Bench 跨方言评估汇总 (RQ4)

- 生成时间: 2026-08-28 00:31
- 模型: openai/whisper-large-v3 | adapter: runs/qlora_full/best_adapter | 仅推理

## 1. corpus_CER 汇总
| 条件 | 数据集 | 条目数 | corpus_CER | 备注 |
|---|---|---:|---:|---|
| a) zeroshot | wu_bench_mandarin | 500 | 0.8517 | preview(500) |
| b) qlora_full | wu_bench_mandarin | 500 | 1.0493 | preview(500) |
| a) zeroshot | wu_bench_mandarin | 3000 | 0.8493 |  |
| b) qlora_full | wu_bench_mandarin | 3000 | 1.0883 |  |
| c) zeroshot | wu_bench_dialect | 4851 | 0.8123 |  |
| d) qlora_full | wu_bench_dialect | 4851 | 1.0647 |  |

## 2. mean_utt_CER 与推理耗时(耗时由 summary.json 时间戳差分估算,首个无起点标记)
| 条件×数据集 | mean_utt_CER | 估算耗时 |
|---|---:|---:|
| zeroshot × wu_bench_mandarin(500) | 0.8506 | n/a(首个) |
| qlora_full × wu_bench_mandarin(500) | 1.0421 | 4.0 min |
| zeroshot × wu_bench_mandarin | 0.8489 | 26.7 min |
| qlora_full × wu_bench_mandarin | 1.0697 | 28.0 min |
| zeroshot × wu_bench_dialect | 0.8181 | 39.4 min |
| qlora_full × wu_bench_dialect | 1.0668 | 42.6 min |

## 3. mandarin 全量抽样对照(按 qlora_full 每条CER排序:5低+5高)
| # | full_CER | 参考(普通话) | zeroshot输出 | qlora_full输出 |
|---|---:|---|---|---|
| 低1 | 0.000 | 他说 | 他说 | 他说 |
| 低2 | 0.000 | 嗯 | 嗯 | 嗯 |
| 低3 | 0.000 | 小姑娘 | 小小姑娘 | 小姑娘 |
| 低4 | 0.000 | 啊 | 哎 | 啊 |
| 低5 | 0.000 | 嗯 | 嗯 | 嗯 |
| 高1 | 11.842 | 你说说看哎急死哎你看看哦观前街上黄天源 | 你看看你看看我看你看我看你看我看你看 | 内考考学上考考考考学学学学学学学学学学学学学学学学学学学学学学学学学学学学学学学 |
| 高2 | 13.133 | 什么十块啊明明是五十块你看错了 | 傻傻快的明明是傻快我愧出他了 | 傻傻快明明傻傻快快快快快快快快快快快快快快快快快快快快快快快快快快快快快快快快快 |
| 高3 | 13.364 | 啊你可以你可以同时来啊 | 都好懂事 | 老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老老 |
| 高4 | 23.429 | 啊那么做生意末 | 那么这三年嘛 | 啊自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自自 |
| 高5 | 28.500 | 他又笑了起来 | 也有小了起来 | 热小的吃得得得得得得得得得得得得得得得得得得得得得得得得得得得得得得得得得得得得 |

## 4. 输出含典型吴语用字的条目占比(诊断:微调是否倾向输出方言字)
典型字/词: 阿拉 伊拉 勿 辣 交关 物事 欢喜 辰光

| 模型 (mandarin全量) | 含吴语字条目 | 占比 |
|---|---:|---:|
| zeroshot | 62/3000 | 2.1% |
| qlora_full | 42/3000 | 1.4% |
_注:辣/欢喜等在普通话中亦常见,为粗略启发式指标。_

## 5. 启东话测试集既有结果(对照)
| 模型 | corpus_CER |
|---|---:|
| zeroshot | 0.8056 |
| qlora_full | 0.5886 |
