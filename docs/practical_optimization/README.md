# Practical Optimization

大規模モデルの学習を実際に速く・安く回すための 3 分野をまとめたディレクトリです。対象は 134 件で、複数分野に属する論文があるため分野別件数の合計は 147 件になります。全体の目次は [../README.md](../README.md)、全件カタログは [../00_catalog.md](../00_catalog.md) を参照してください。

| # | ドキュメント | 対象 | 件数 |
|---|---|---|---|
| 1 | [01_critical_batch_size.md](01_critical_batch_size.md) | クリティカルバッチサイズ（CBS）、大バッチ学習、勾配ノイズスケール、バッチサイズ/学習率スケーリング則 | 61（ユニーク 56 本） |
| 2 | [02_low_precision_and_muon.md](02_low_precision_and_muon.md) | 低精度学習（FP8/FP4/NVFP4/量子化）、Muon と直交化・スペクトル系オプティマイザ | 69（ユニーク 66 本） |
| 3 | [03_semi_synchronous_training.md](03_semi_synchronous_training.md) | セミシンクロナス学習（Local SGD、DiLoCo 系、外部オプティマイザ、非同期・gossip） | 17（+関連 2 件） |

## 3 分野の時系列

公開年ごとの件数（重複登録を含む）:

| 公開年 | クリティカルバッチサイズ | 低精度・Muon | セミシンクロナス |
|---|---|---|---|
| 2017 | 3 | 0 | 0 |
| 2018 | 5 | 1 | 1 |
| 2019 | 3 | 0 | 0 |
| 2020 | 10 | 1 | 1 |
| 2021 | 4 | 2 | 2 |
| 2022 | 3 | 2 | 1 |
| 2023 | 1 | 0 | 1 |
| 2024 | 4 | 2 | 0 |
| 2025 | 16 | 26 | 5 |
| 2026 | 9 | 27 | 6 |
| 不明 | 3 | 8 | 0 |

- **クリティカルバッチサイズ**: Shallue et al.（[#16](https://github.com/Hiroki11x/Papers/issues/16)）が示した 3 領域構造と、McCandlish et al.（[#8](https://github.com/Hiroki11x/Papers/issues/8)）の勾配ノイズスケールが出発点です。2020 年前後に SDE とスケーリングルールの研究が続き、2024〜2026 年は LLM 時代の問いに移りました。CBS は主にデータ量でスケールするのか（[#390](https://github.com/Hiroki11x/Papers/issues/390)、[#391](https://github.com/Hiroki11x/Papers/issues/391)）、オプティマイザで CBS は変わるのか（[#456](https://github.com/Hiroki11x/Papers/issues/456)、[#553](https://github.com/Hiroki11x/Papers/issues/553)）、RL でのバッチ拡大は有効か（[#572](https://github.com/Hiroki11x/Papers/issues/572)）といった問いです。
- **低精度・Muon**: 量子化は CNN の推論圧縮から始まり、FP8 での 20T トークン事前学習（[#442](https://github.com/Hiroki11x/Papers/issues/442)）、NVFP4 での 4bit 事前学習（[#444](https://github.com/Hiroki11x/Papers/issues/444)）へと進みました。行列オプティマイザは Shampoo（[#451](https://github.com/Hiroki11x/Papers/issues/451)）を源流とし、Muon の登場後の 2025 年後半から、理論解析・改良・ベンチマーク・ノルム制約の研究が急増しています。中心的な論点は「Muon の AdamW に対する優位はスケールで消えるか」で、[#432](https://github.com/Hiroki11x/Papers/issues/432) と [#474](https://github.com/Hiroki11x/Papers/issues/474) で結論が異なります。
- **セミシンクロナス**: Local SGD の汎化と大規模での限界（[#9](https://github.com/Hiroki11x/Papers/issues/9)、[#70](https://github.com/Hiroki11x/Papers/issues/70)）から始まります。DiLoCo 以降は外部 Nesterov による最適化アルゴリズムとしての再解釈（[#534](https://github.com/Hiroki11x/Papers/issues/534)、[#536](https://github.com/Hiroki11x/Papers/issues/536)）へ進み、2026 年には同期間隔の適応化（[#573](https://github.com/Hiroki11x/Papers/issues/573)）、gossip 化（[#530](https://github.com/Hiroki11x/Papers/issues/530)）、遅延への頑健性とオプティマイザの関係（[#557](https://github.com/Hiroki11x/Papers/issues/557)）が扱われています。

## 3 分野のつながり

- **バッチサイズとオプティマイザ**: [#456](https://github.com/Hiroki11x/Papers/issues/456) では完全 Gauss-Newton 法が CBS を大きく広げる一方、AdamW と Muon はともに約 12M トークンで頭打ちになります。[#553](https://github.com/Hiroki11x/Papers/issues/553) は、Muon（SignSVD）の前処理効果が大バッチでのみ現れることを示しています。オプティマイザ比較の結論はバッチサイズで逆転しうる、というのが両ドキュメントに共通する整理です（[#432](https://github.com/Hiroki11x/Papers/issues/432) と [#433](https://github.com/Hiroki11x/Papers/issues/433) の食い違い）。
- **同期間隔とバッチサイズ**: Local SGD / DiLoCo の内部ステップ数 $H$ とワーカー数 $M$ は実効バッチサイズと絡むため、CBS の議論と直結します。
- **Muon × 分散 × 低精度**: Muon の直交化の通信コスト（Dion、MuonBP [#458](https://github.com/Hiroki11x/Papers/issues/458)）、遅延への頑健性（[#557](https://github.com/Hiroki11x/Papers/issues/557)）、量子化耐性（[#521](https://github.com/Hiroki11x/Papers/issues/521)）、INT8 通信（[#573](https://github.com/Hiroki11x/Papers/issues/573)）が 3 分野の交差点にあります。

## 関連する misc トピック

- Muon 以外のオプティマイザ（2 次法、適応的手法、ベンチマーク）: [../misc/07_optimizer_design.md](../misc/07_optimizer_design.md)
- 学習率スケジュール、ウォームアップ、weight decay: [../misc/08_lr_schedule_weight_decay.md](../misc/08_lr_schedule_weight_decay.md)
- スケーリング則全般: [../misc/09_scaling_laws.md](../misc/09_scaling_laws.md)
- 大バッチの汎化ギャップと平坦性: [../misc/03_loss_landscape_sharpness.md](../misc/03_loss_landscape_sharpness.md)、[../misc/06_sgd_dynamics_theory.md](../misc/06_sgd_dynamics_theory.md)

## 注意点

- [#505](https://github.com/Hiroki11x/Papers/issues/505)（ARO）は、ノートの内容（バッチランプと GNS の理論）が論文題名（行列最適化）と一致していないため、原論文での確認が必要です。
