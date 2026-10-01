# misc

[Practical Optimization](../practical_optimization/README.md) の重点 3 分野に入らない 438 件を、12 トピックに分類したディレクトリです。各論文は主トピック 1 つに属し、他トピックとの関係は各ドキュメントの「関連論文」節にリンクしています。全体の目次は [../README.md](../README.md)、全件カタログは [../00_catalog.md](../00_catalog.md) を参照してください。

## トピック一覧

| # | ドキュメント | 主な内容 | 件数（ユニーク） | 論文の公開期間 |
|---|---|---|---|---|
| 01 | [01_ood_generalization.md](01_ood_generalization.md) | OOD 汎化、ドメイン汎化、分布シフト、敵対的・自然な頑健性 | 74（72） | 2017-06〜2026-02 |
| 02 | [02_calibration_uncertainty.md](02_calibration_uncertainty.md) | キャリブレーション、不確実性推定、ベイズ深層学習、ディープアンサンブル | 44（42） | 2017-06〜2026-08 |
| 03 | [03_loss_landscape_sharpness.md](03_loss_landscape_sharpness.md) | 損失地形、シャープネス、SAM、ヘシアン/フィッシャースペクトル、Edge of Stability | 31（29） | 2018-02〜2026-07 |
| 04 | [04_generalization_implicit_bias.md](04_generalization_implicit_bias.md) | 汎化理論、暗黙的バイアス、過剰パラメータ化、NTK/μP、Grokking、特徴学習 | 59（57） | 2016-12〜2025-09 |
| 05 | [05_regularization_augmentation_compression.md](05_regularization_augmentation_compression.md) | 正則化、データ拡張、モデル圧縮、知識蒸留、正規化層の理論 | 27（26） | 2019-07〜2023-01 |
| 06 | [06_sgd_dynamics_theory.md](06_sgd_dynamics_theory.md) | SGD の SDE 近似、ノイズ構造、鞍点脱出、収束解析、モメンタム理論、初期化 | 29（29） | 2014-06〜2026-09 |
| 07 | [07_optimizer_design.md](07_optimizer_design.md) | 2 次法（K-FAC 等）、適応的手法、学習型オプティマイザ、オプティマイザ比較 | 49（48） | 2015-03〜2026-06 |
| 08 | [08_lr_schedule_weight_decay.md](08_lr_schedule_weight_decay.md) | 学習率スケジュール（WSD・cooldown・schedule-free）、ウォームアップ、weight decay と実効学習率 | 22（22） | 2020-04〜2026-08 |
| 09 | [09_scaling_laws.md](09_scaling_laws.md) | 汎化誤差・視覚・転移・蒸留・推論コストのスケーリング則、データプルーニング | 16（16） | 2019-09〜2026-02 |
| 10 | [10_gan_minimax.md](10_gan_minimax.md) | GAN の学習と安定化、ミニマックス最適化、ゲームダイナミクス | 20（18） | 2017-03〜2022-10 |
| 11 | [11_llm_architecture_reasoning_safety.md](11_llm_architecture_reasoning_safety.md) | LLM のアーキテクチャ、RL による推論、事前学習データ、幻覚、安全性、エージェント | 35（33） | 2021-07〜2026-09 |
| 12 | [12_continual_rl_misc.md](12_continual_rl_misc.md) | 継続学習、強化学習、その他のトピック | 32（31） | 1990-01〜2026-08 |

件数は issue 数で、括弧内は同一論文の重複登録を除いた本数です（10 は内容未記入の [#93](https://github.com/Hiroki11x/Papers/issues/93) も除外）。

## トピックのまとまり

- **汎化と頑健性**（01, 02, 04, 05）: 2020〜2022 年に集中して読んだ領域です。分布シフト下の汎化と不確実性（01・02）、なぜ過剰パラメータ化したネットワークが汎化するのか（04）、正則化・データ拡張・蒸留による汎化の改善（05）を扱います。
- **最適化の理論と地形**（03, 06）: シャープネスと汎化の関係、SGD ノイズの構造、Edge of Stability など、オプティマイザの挙動を理論的に理解する研究です。大バッチ学習の汎化ギャップは [../practical_optimization/01_critical_batch_size.md](../practical_optimization/01_critical_batch_size.md) と重なります。
- **最適化手法とハイパーパラメータ**（07, 08, 09）: オプティマイザの設計と比較、学習率スケジュールと weight decay、スケーリング則です。2025 年以降は LLM 事前学習の文脈のノートが増えており、Muon などの行列オプティマイザは [../practical_optimization/02_low_precision_and_muon.md](../practical_optimization/02_low_precision_and_muon.md) で扱っています。
- **その他の領域**（10, 11, 12）: GAN とミニマックス最適化（2021〜2022 年の輪読会のノートを含む）、LLM のアーキテクチャ・推論・安全性（2025 年 9 月以降が中心）、継続学習・強化学習などです。

## 時期の偏り

issue 作成年（読んだ年）ごとの件数は [../00_catalog.md](../00_catalog.md) の「issue 作成年 × トピック」にあります。01〜05 と 10 は 2020〜2022 年に、08・11 は 2025〜2026 年に集中しています。06・07 は 2020〜2022 年が中心ですが、2025〜2026 年にも LLM 時代の論文が加わっています（06 は 6 件、07 は 13 件）。09 は 2025 年に 9 件と、後半の時期の比重が大きいトピックです。12 は 2022 年の 17 件が中心です。
