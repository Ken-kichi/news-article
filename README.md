# news-article

ニュース記事の原稿から音声・画像を生成し、ショート動画を自動で組み立てるスクリプトです。LangGraph で構築した 6 ノードのワークフローを `uv run main.py <開始日> <終了日>` で実行します。

## セットアップ

1. **Python 環境**

   ```bash
   uv sync
   ```

2. **環境変数 (`.env`)**

   - `AZURE_TEXT_API_KEY` / `AZURE_TEXT_ENDPOINT`
   - `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION`
   - `AZURE_IMAGE_KEY` / `AZURE_IMAGE_ENDPOINT`
   - 任意: `JP_FONT_PATH`（日本語フォントパス）

3. **ローカルディレクトリ**

   ```
   article/        … 20240101_title.md 形式の原稿
   movie/          … ランダム切り出し元となる縦向き/横向き動画
   output/         … 生成物（自動作成）
   ```

## 使い方

```bash
uv run main.py 20260209 20260210
```

実行後、`output/20260209_20260210/` に以下が保存されます。

- `audio_*.wav` … Azure Speech によるナレーション
- `script_*.sbv` … SBV形式の字幕スクリプト（YouTubeへそのままアップロード可能）
- `image_*.png` … Azure OpenAI Image によるサムネイル
- `final_youtube_short.mp4` … 最終動画（画像→5 秒ごとに movie から抽出した映像へ切り替え）
- `node_logs.jsonl` … 各ノードのアウトプットログ
- `youtube_meta.txt` … YouTube タイトル・説明文・ハッシュタグのセット
- `youtube_thumbnail_<期間>.png` … YouTube 用サムネイル画像

## ワークフロー概要

1. **fetch_articles**  
   `article/` から指定期間の Markdown を読み込み、GPT で 500 文字以内の口語原稿に要約。

2. **generate_audio_assets**  
   Azure Speech で記事ごとの音声 `audio_*.wav` とナレーション台本 `script_*.txt` を生成。

3. **generate_visual_assets**  
   GPT で英語プロンプトを作り、Azure Image (FLUX) で縦長アートを描画。

4. **generate_thumbnail**  
   動画全体のトーンに合わせた 16:9 のサムネイルを FLUX で生成し、`youtube_thumbnail_<期間>.png` として保存。

5. **create_video**  
   生成画像＋`movie/` のフリー素材を組み合わせ、音声を連結して `final_youtube_short.mp4` を出力。

6. **generate_youtube_metadata**  
   動画タイトル／説明文／ハッシュタグを `youtube_meta.txt` に書き出し、そのまま YouTube へ貼り付け可能にする。

## ログ

`node_logs.jsonl` は JSON Lines 形式で、各ノードの入出力（記事タイトル、生成ファイル、動画に使った元動画と秒数など）を追跡できます。
