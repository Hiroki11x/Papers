# 論文ノート サーベイ集（2026-09-30 時点）

このディレクトリは、GitHub Issues に記録した論文ノート全 572 件（[#1](https://github.com/Hiroki11x/Papers/issues/1)〜[#573](https://github.com/Hiroki11x/Papers/issues/573)）を読み直し、分野ごとに再構成したドキュメントです。特に次の 3 分野を重点的にまとめています。

| # | ドキュメント | 対象 | 件数 |
|---|---|---|---|
| 0 | [00_catalog.md](00_catalog.md) | 全 572 件のカタログ（分野・採択先・公開年の集計、その他分野のラベル別一覧） | 572 |
| 1 | [01_critical_batch_size.md](01_critical_batch_size.md) | クリティカルバッチサイズ・大バッチ学習・勾配ノイズスケール・バッチサイズ/学習率スケーリング則 | 61 |
| 2 | [02_low_precision_and_muon.md](02_low_precision_and_muon.md) | 低精度学習（FP8/FP4/NVFP4/量子化）と Muon・直交化/スペクトル系オプティマイザ | 69 |
| 3 | [03_semi_synchronous_training.md](03_semi_synchronous_training.md) | セミシンクロナス学習（Local SGD、DiLoCo 系、外部オプティマイザ、非同期・gossip） | 17（+追加2） |

各分野ドキュメントは共通して、背景と基本概念、時系列ナラティブ、Mermaid タイムライン、サブトピック別整理、論文一覧表（公開年月・採択先・根拠つき）、採択先別集計、各論文の詳細まとめ、横断的な知見・未解決問題、関連論文の順で構成されています。

## 3 分野の時系列（全体像）

公開年ごとの件数（重複登録を含む。詳細は [00_catalog.md](00_catalog.md)）:

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

ノートの読み方は大きく二つの時期に分かれます。2020〜2022 年は大バッチ学習・汎化・OOD の基礎研究を幅広く読んでいた時期で、2025〜2026 年は LLM 事前学習の効率化（バッチサイズ、Muon、低精度、低通信分散学習）に集中しています。

- **クリティカルバッチサイズ**: 2018 年の Shallue et al.（[#16](https://github.com/Hiroki11x/Papers/issues/16)）が示した 3 領域構造と McCandlish et al.（[#8](https://github.com/Hiroki11x/Papers/issues/8)）の勾配ノイズスケールが出発点です。2020 年前後に SDE とスケーリングルールの研究が続き、2024〜2026 年は LLM 時代の問いに移りました。CBS は損失ではなく主にデータ量でスケールするのか（[#390](https://github.com/Hiroki11x/Papers/issues/390)、[#391](https://github.com/Hiroki11x/Papers/issues/391)）、オプティマイザで CBS はどう変わるのか（[#456](https://github.com/Hiroki11x/Papers/issues/456)、[#553](https://github.com/Hiroki11x/Papers/issues/553)）、RL でのバッチ拡大は有効か（[#572](https://github.com/Hiroki11x/Papers/issues/572)）、といった問いです。
- **低精度・Muon**: 量子化は CNN の推論圧縮から始まり、FP8 での 20T トークン事前学習（[#442](https://github.com/Hiroki11x/Papers/issues/442)）、NVFP4 での 4bit 事前学習（[#444](https://github.com/Hiroki11x/Papers/issues/444)）へと進みました。行列オプティマイザは Shampoo（[#451](https://github.com/Hiroki11x/Papers/issues/451)）を源流とし、Muon の登場後の 2025 年後半から、理論解析・改良・ベンチマーク・ノルム制約の研究が急増しています。中心的な論点は「Muon の AdamW に対する優位はスケールで消えるか」です（[#432](https://github.com/Hiroki11x/Papers/issues/432) と [#474](https://github.com/Hiroki11x/Papers/issues/474) で結論が異なります）。
- **セミシンクロナス**: Local SGD の汎化と大規模での限界（[#9](https://github.com/Hiroki11x/Papers/issues/9)、[#70](https://github.com/Hiroki11x/Papers/issues/70)）から始まります。DiLoCo 以降は「外部 Nesterov による最適化アルゴリズム」としての再解釈（[#534](https://github.com/Hiroki11x/Papers/issues/534)、[#536](https://github.com/Hiroki11x/Papers/issues/536)）へ進み、2026 年には同期間隔の適応化（[#573](https://github.com/Hiroki11x/Papers/issues/573)）、gossip 化（[#530](https://github.com/Hiroki11x/Papers/issues/530)）、遅延への頑健性とオプティマイザの関係（[#557](https://github.com/Hiroki11x/Papers/issues/557)）が扱われています。

## 3 分野のつながり

- **バッチサイズとオプティマイザ**: Muon や二次最適化は CBS を押し上げるとされ、オプティマイザ比較の結論はバッチサイズで逆転しうる、とノートでは整理しています（[#432](https://github.com/Hiroki11x/Papers/issues/432) と [#433](https://github.com/Hiroki11x/Papers/issues/433) の食い違い）。
- **同期間隔とバッチサイズ**: Local SGD / DiLoCo の内部ステップ数 $H$ とワーカー数 $M$ は実効バッチサイズと絡むため、CBS の議論と直結します。
- **Muon × 分散 × 低精度**: Muon の直交化の通信コスト（Dion、MuonBP [#458](https://github.com/Hiroki11x/Papers/issues/458)）、遅延への頑健性（[#557](https://github.com/Hiroki11x/Papers/issues/557)）、量子化耐性（[#521](https://github.com/Hiroki11x/Papers/issues/521)）、INT8 通信（[#573](https://github.com/Hiroki11x/Papers/issues/573)）が 3 分野の交差点にあります。

## 作成方法と注意点

- 全 issue の本文とコメントを 8 分割して並列に読み、各論文の分類・公開年月・採択先・要約を抽出しました。そのうえで分野ごとに元ノートを読み直してドキュメントを書いています。
- **採択先の根拠**は、issue記載（ノート本文・リンク）、arXivコメント（arXiv API の comment / journal_ref 欄）、既知情報（論文の既知の採択情報）、不明の 4 種類で示しています。「不明」は採択の根拠が見つからなかったものをプレプリント扱いにしたもので、未採択と確定したわけではありません。特に 2026 年の論文は採択状況が今後変わる可能性があります。
- **公開年月**は arXiv の初回投稿月です。issue の作成日（読んだ日）とは別に扱っています。
- 同一論文を二重に登録した issue（例: [#16](https://github.com/Hiroki11x/Papers/issues/16)/[#403](https://github.com/Hiroki11x/Papers/issues/403)、[#426](https://github.com/Hiroki11x/Papers/issues/426)/[#431](https://github.com/Hiroki11x/Papers/issues/431)）は件数に重複して含め、各ドキュメント内で明記しています。
- [#505](https://github.com/Hiroki11x/Papers/issues/505)（ARO）はノートの要約内容（バッチランプと GNS の理論）が論文題名（行列最適化）と一致していないため、原論文での確認が必要です。
