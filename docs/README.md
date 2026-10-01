# 論文ノート サーベイ集（2026-09-30 時点）

このディレクトリは、GitHub Issues に記録した論文ノート全 572 件（[#1](https://github.com/Hiroki11x/Papers/issues/1)〜[#573](https://github.com/Hiroki11x/Papers/issues/573)）を読み直し、分野ごとに再構成したドキュメントです。全件を次の 2 区分・15 トピックのいずれかに分類しています。

- **[Practical Optimization](practical_optimization/README.md)**: 重点 3 分野。クリティカルバッチサイズ、低精度学習と Muon、セミシンクロナス学習（134 件。複数分野に属する論文があるため、分野別件数の合計は 147 件）。
- **[misc](misc/README.md)**: それ以外の 12 トピック。OOD 汎化、キャリブレーション、損失地形、汎化理論、オプティマイザ設計、スケーリング則、LLM など（438 件。各論文は主トピック 1 つに属する）。
- **[00_catalog.md](00_catalog.md)**: 全 572 件のカタログ。トピック別・公開年別・issue 作成年別・採択先別の集計と、トピックごとの全件一覧。

## ディレクトリ構成

```
docs/
├── README.md                  # このファイル（全体の目次）
├── 00_catalog.md              # 全572件のカタログ（集計・採択先別一覧・トピック別一覧）
├── practical_optimization/    # 重点3分野
│   ├── README.md
│   ├── 01_critical_batch_size.md
│   ├── 02_low_precision_and_muon.md
│   └── 03_semi_synchronous_training.md
└── misc/                      # その他12トピック
    ├── README.md
    ├── 01_ood_generalization.md
    ├── 02_calibration_uncertainty.md
    ├── 03_loss_landscape_sharpness.md
    ├── 04_generalization_implicit_bias.md
    ├── 05_regularization_augmentation_compression.md
    ├── 06_sgd_dynamics_theory.md
    ├── 07_optimizer_design.md
    ├── 08_lr_schedule_weight_decay.md
    ├── 09_scaling_laws.md
    ├── 10_gan_minimax.md
    ├── 11_llm_architecture_reasoning_safety.md
    └── 12_continual_rl_misc.md
```

## トピック一覧

| 区分 | ドキュメント | 主な内容 | 件数 |
|---|---|---|---|
| Practical Optimization | [01_critical_batch_size.md](practical_optimization/01_critical_batch_size.md) | クリティカルバッチサイズ、大バッチ学習、勾配ノイズスケール、バッチサイズ/学習率スケーリング則 | 61 |
| Practical Optimization | [02_low_precision_and_muon.md](practical_optimization/02_low_precision_and_muon.md) | 低精度学習（FP8/FP4/NVFP4/量子化）、Muon と直交化・スペクトル系オプティマイザ | 69 |
| Practical Optimization | [03_semi_synchronous_training.md](practical_optimization/03_semi_synchronous_training.md) | Local SGD、DiLoCo 系、外部オプティマイザ、非同期・gossip 学習 | 17（+関連2） |
| misc | [01_ood_generalization.md](misc/01_ood_generalization.md) | OOD 汎化、ドメイン汎化、分布シフト、頑健性 | 74 |
| misc | [02_calibration_uncertainty.md](misc/02_calibration_uncertainty.md) | キャリブレーション、不確実性推定、ベイズ深層学習、アンサンブル | 44 |
| misc | [03_loss_landscape_sharpness.md](misc/03_loss_landscape_sharpness.md) | 損失地形、シャープネス、SAM、ヘシアン/フィッシャースペクトル、Edge of Stability | 31 |
| misc | [04_generalization_implicit_bias.md](misc/04_generalization_implicit_bias.md) | 汎化理論、暗黙的バイアス、過剰パラメータ化、NTK/μP、Grokking | 59 |
| misc | [05_regularization_augmentation_compression.md](misc/05_regularization_augmentation_compression.md) | 正則化、データ拡張、モデル圧縮、知識蒸留、正規化層 | 27 |
| misc | [06_sgd_dynamics_theory.md](misc/06_sgd_dynamics_theory.md) | SGD の SDE 近似、ノイズ構造、鞍点脱出、収束解析、モメンタム理論 | 29 |
| misc | [07_optimizer_design.md](misc/07_optimizer_design.md) | 2 次法、適応的手法、学習型オプティマイザ、オプティマイザ比較・ベンチマーク | 49 |
| misc | [08_lr_schedule_weight_decay.md](misc/08_lr_schedule_weight_decay.md) | 学習率スケジュール（WSD・cooldown・schedule-free）、ウォームアップ、weight decay | 22 |
| misc | [09_scaling_laws.md](misc/09_scaling_laws.md) | 汎化・転移・蒸留・推論コストのスケーリング則、データプルーニング | 16 |
| misc | [10_gan_minimax.md](misc/10_gan_minimax.md) | GAN の学習と安定化、ミニマックス最適化、ゲームダイナミクス | 20 |
| misc | [11_llm_architecture_reasoning_safety.md](misc/11_llm_architecture_reasoning_safety.md) | LLM のアーキテクチャ、RL による推論、事前学習データ、安全性、エージェント | 35 |
| misc | [12_continual_rl_misc.md](misc/12_continual_rl_misc.md) | 継続学習、強化学習、その他 | 32 |

各トピックのドキュメントは、背景、時系列の流れ（Mermaid タイムラインつき）、サブトピック別の整理、論文一覧表（公開年月・採択先・根拠）、採択先別の集計、各論文の詳細、横断的な知見と未解決問題、関連論文の順で構成しています。

## 全体の時系列

issue の作成時期（読んだ時期）で見ると、ノートは大きく 2 つの時期に分かれます。

- **2020〜2022 年（360 件）**: 大バッチ学習と汎化、OOD 汎化、キャリブレーション、損失地形、汎化理論、GAN など、深層学習の基礎研究を幅広く読んでいた時期です。misc の大半はこの時期の論文です。2023 年は 13 件で、2024 年の登録はありません。
- **2025〜2026 年（199 件）**: LLM 事前学習の効率化に集中した時期です。バッチサイズとデータ量のスケーリング、Muon などの行列オプティマイザ、FP8/NVFP4 の低精度学習、DiLoCo 系の低通信分散学習が中心で、学習率スケジュール（WSD・cooldown）や LLM の推論・安全性のノートもこの時期に増えています。

公開年・issue 作成年ごとのトピック別件数は [00_catalog.md](00_catalog.md) にまとめています。各分野の詳細な時系列は個別ドキュメントのタイムラインを参照してください。

## トピック間のつながり

- **バッチサイズとオプティマイザ**（[CBS](practical_optimization/01_critical_batch_size.md) × [Muon](practical_optimization/02_low_precision_and_muon.md) × [オプティマイザ設計](misc/07_optimizer_design.md)）: [#456](https://github.com/Hiroki11x/Papers/issues/456) では完全 Gauss-Newton 法が CBS を大きく広げる一方、AdamW と Muon はともに約 12M トークンで頭打ちになります。[#553](https://github.com/Hiroki11x/Papers/issues/553) は、Muon（SignSVD）の前処理効果が大バッチでのみ現れることを示しています。オプティマイザ比較の結論はバッチサイズやスケールで変わりうるため、[#432](https://github.com/Hiroki11x/Papers/issues/432) と [#474](https://github.com/Hiroki11x/Papers/issues/474) では Muon の優位性について異なる結論が出ています。
- **バッチサイズと学習率スケジュール**（[CBS](practical_optimization/01_critical_batch_size.md) × [学習率スケジュール](misc/08_lr_schedule_weight_decay.md) × [スケーリング則](misc/09_scaling_laws.md)）: バッチサイズ/学習率のスケーリング則、WSD とバッチランプ、weight decay と実効学習率は、LLM 事前学習のハイパーパラメータ設計として一続きの問題です。
- **同期間隔とバッチサイズ**（[セミシンクロナス](practical_optimization/03_semi_synchronous_training.md) × [CBS](practical_optimization/01_critical_batch_size.md)）: Local SGD / DiLoCo の内部ステップ数とワーカー数は実効バッチサイズと絡むため、CBS の議論と直結します。
- **Muon × 分散 × 低精度**: Muon の直交化の通信コスト（MuonBP [#458](https://github.com/Hiroki11x/Papers/issues/458)）、遅延への頑健性（[#557](https://github.com/Hiroki11x/Papers/issues/557)）、量子化耐性（[#521](https://github.com/Hiroki11x/Papers/issues/521)）、INT8 通信（[#573](https://github.com/Hiroki11x/Papers/issues/573)）が重点 3 分野の交差点にあります。
- **平坦性・汎化・SGD ノイズ**（[損失地形](misc/03_loss_landscape_sharpness.md) × [汎化理論](misc/04_generalization_implicit_bias.md) × [SGD ダイナミクス](misc/06_sgd_dynamics_theory.md)）: 大バッチ学習の汎化ギャップ、SGD の暗黙的正則化、シャープネスと汎化の関係は、この 3 トピックと CBS ドキュメントにまたがって議論されています。
- **分布シフトと不確実性**（[OOD 汎化](misc/01_ood_generalization.md) × [キャリブレーション](misc/02_calibration_uncertainty.md)）: 分布シフト下でのキャリブレーションやアンサンブルの頑健性は両トピックで扱っています。

## 作成方法と検証

- **抽出と分類**: 全 issue の本文とコメントを分割して並列に読み、各論文の分類・公開年月・採択先・要約を抽出しました。misc のトピック分類は別のエージェントがレビューし、4 件の分類を修正しています。
- **採択先の検証**: arXiv API のコメント欄、Semantic Scholar、Web（OpenReview・proceedings・DOI 等）で採択先を照合しました。疑わしい 96 件を個別に確認して 27 件を修正し、その後の各ドキュメントのレビューでもさらに十数件を修正しています。
- **ドキュメントのレビュー**: 15 本すべてを、執筆とは別のレビューエージェントが元ノートと照合して修正しました。最後に、ドキュメント間で同じ論文の記述が矛盾していないか、採択先がレコードと一致しているか、リンクが切れていないかを横断的に確認しています。

### 採択先の根拠の読み方

| 根拠 | 意味 |
|---|---|
| issue記載 | ノート本文やリンクに採択先が書かれている |
| arXivコメント | arXiv の comment / journal_ref 欄に採択先が書かれている |
| Semantic Scholar確認 | Semantic Scholar の venue 情報で確認した |
| Web確認 | OpenReview・proceedings・DOI ページ等で確認した |
| 不明 | 採択の根拠が見つからず「arXiv（プレプリント）」扱いにした（未採択と確定したわけではない） |

特に 2026 年の論文は、今後採択状況が変わる可能性があります。**公開年月**は arXiv の初回投稿月で、issue の作成日（読んだ日）とは別に扱っています。

## 注意点（要確認の issue）

- **重複登録**: 同一論文を二重に登録した issue（例: [#16](https://github.com/Hiroki11x/Papers/issues/16)/[#403](https://github.com/Hiroki11x/Papers/issues/403)、[#426](https://github.com/Hiroki11x/Papers/issues/426)/[#431](https://github.com/Hiroki11x/Papers/issues/431)）は両方を件数に含め、各ドキュメントで明記しています。[#30](https://github.com/Hiroki11x/Papers/issues/30)/[#104](https://github.com/Hiroki11x/Papers/issues/104) は同じ arXiv エントリの v1 と、内容を大きく改訂した v2 のノートなので、別々に扱っています。
- [#505](https://github.com/Hiroki11x/Papers/issues/505)（ARO）は、ノートの内容（バッチランプと GNS の理論）が論文題名（行列最適化）と一致していないため、原論文での確認が必要です。
- [#93](https://github.com/Hiroki11x/Papers/issues/93)（Minmax Optimization）は本文が空のプレースホルダーです。
