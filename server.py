import os
import httpx
import logging
import json
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr, ValidationError

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# FastAPIアプリケーションの初期化
app = FastAPI()

# CORSミドルウェアの設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

# --- Function Callingの仕様定義 ---
# AIが利用できる関数（ツール）の定義
# https://note.com/vitaactiva/n/ncee4997bbb63
tool_app_rag = {
    "type": "function",
    "name": "appRAG",
    "description": "○○薬局のQ&Aデータベースを検索し、ユーザーの質問に合致する回答を見つけます。ユーザーの口語的な質問から、検索に最適化された簡潔な質問文を生成して引数として使用します。",
    "parameters": {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "検索に使用する、ユーザーの質問の意図を正確に反映した、正規化された日本語の質問文。"
            }
        },
        "required": ["search_query"]
    }
}
# --- システムプロンプトの定義 ---
# AIの基本的な役割や性格を定義
system_prompt = """
# 指示
あなたは「〇〇薬局」の薬剤師アシスタントとして、親切・丁寧・正確にお客様の質問に回答します。

# ルール
- 回答は必ず`appRAG`関数で得た情報のみに基づき、自己の知識や推測は使用禁止です。
- `appRAG`で情報が見つからない場合、「申し訳ありません、お尋ねの件については分かりかねます。」と回答してください。
- 医療相談や診断に関する質問には、決して自分で判断せず、次の通りに回答し電話を促してください：「その件については専門の薬剤師が直接ご説明しますので、お手数ですがお電話ください。」
"""

# 1. Offer/Answerを中継するプロキシエンドポイント
@app.post("/api/realtime-proxy")
async def realtime_proxy(request: Request):
    """
    フロントエンドからSDP Offerを受け取り、OpenAI APIに中継して
    SDP Answerを返すプロキシ。セッション開始時にツールとプロンプトを設定する。
    """
    try:
        # 1) フロントエンドから受け取ったOffer SDP
        offer_sdp = (await request.body()).decode('utf-8')
        logging.info(f"Received Offer SDP (first 50 chars): {offer_sdp[:50]}...")

        # 非同期HTTPクライアントの準備
        async with httpx.AsyncClient() as client:
            # 2) /v1/realtime/sessions でエフェメラルキーを取得
            # https://note.com/npaka/n/nf9cab7ea954e
            # https://platform.openai.com/docs/api-reference/realtime-sessions/create
            ephemeral_resp = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "OpenAI-Beta": "realtime=v1",
                },
                json={
                    "model": "gpt-4o-mini-realtime-preview-2024-12-17",
                    "instructions": system_prompt,
                    "voice": "shimmer",
                    # "input_audio_transcription": {
                    #     "model": "whisper-1",
                    #     "language": "ja"
                    # },
                    "turn_detection": {
                        "type": "server_vad",
                        "create_response": True,
                        "threshold": 0.8,
                        "silence_duration_ms": 1000
                    },
                    "tools": [tool_app_rag],
                    "temperature": 0.8,
                    "max_response_output_tokens": 500,
                },
                timeout=10,
            )
            ephemeral_resp.raise_for_status()
            ephemeral_data = ephemeral_resp.json()
            ephemeral_key = ephemeral_data.get("client_secret", {}).get("value")

            if not ephemeral_key:
                raise HTTPException(status_code=500, detail="No ephemeral key in response")
            
            logging.info("Successfully received ephemeral key.")

            # 3) /v1/realtime に Offer SDP を送信してAnswer SDPを受け取る
            sdp_resp = await client.post(
                "https://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17",
                headers={
                    "Authorization": f"Bearer {ephemeral_key}",
                    "Content-Type": "application/sdp",
                },
                content=offer_sdp,
                timeout=10,
            )
            sdp_resp.raise_for_status()

            answer_sdp = sdp_resp.text
            logging.info(f"Successfully received Answer SDP (length: {len(answer_sdp)}). Sending to client.")

            # 4) そのままフロントに返す
            return PlainTextResponse(content=answer_sdp)

    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error occurred while contacting OpenAI: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logging.exception("An unexpected error occurred in /api/realtime-proxy")
        raise HTTPException(status_code=500, detail="Error in /api/realtime-proxy")
    

# --- 2. Function Calling実行用のWebSocketエンドポイント (セキュリティチェックを省略) ---

def appRAG(search_query: str) -> str:
    logging.info(f"Executing appRAG for: {search_query}")
    # return f"「{search_query}」に関する質問は、データベースが未実装のため回答できません。"
    return f"営業時間は午前9時から午後10時までです。"

# --- Pydanticモデルによる入力検証 ---
class AppRagArgs(BaseModel):
    search_query: constr(max_length=200)

# --- 関数ディスパッチャ ---
AVAILABLE_FUNCTIONS = {
    "appRAG": appRAG
}
FUNCTION_SCHEMAS = {
    "appRAG": AppRagArgs
}

@app.websocket("/ws/function-call")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established for function calling.")
    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)

            call_id = data.get("call_id")
            function_name = data.get("name")
            arguments_str = data.get("arguments")

            if function_name in AVAILABLE_FUNCTIONS:
                try:
                    arguments = json.loads(arguments_str)
                    schema = FUNCTION_SCHEMAS[function_name]
                    validated_args = schema(**arguments)
                    function_to_call = AVAILABLE_FUNCTIONS[function_name]
                    result = function_to_call(**validated_args.dict())

                    await websocket.send_json({
                        "status": "success",
                        "call_id": call_id,
                        "result": result
                    })
                    logging.info(f"Successfully executed function '{function_name}' and sent result.")

                except ValidationError as e:
                    error_message = f"Invalid arguments for {function_name}: {e}"
                    logging.error(error_message)
                    await websocket.send_json({"status": "error", "call_id": call_id, "message": error_message})
                except Exception as e:
                    error_message = f"Function execution failed for {function_name}: {e}"
                    logging.error(error_message)
                    await websocket.send_json({"status": "error", "call_id": call_id, "message": error_message})
            else:
                error_message = f"Unknown function requested: {function_name}"
                logging.warning(error_message)
                await websocket.send_json({"status": "error", "call_id": call_id, "message": error_message})

    except WebSocketDisconnect:
        logging.info("Client disconnected from WebSocket.")
    except Exception as e:
        logging.error(f"An error occurred in WebSocket: {e}", exc_info=True)