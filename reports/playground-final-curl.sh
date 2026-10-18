curl --request POST 'http://127.0.0.1:18000/v1/systemone' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "model": "lite-omnijev-e4b-dev",
  "state": "用户反馈：新版本导出报告时偶尔卡住，但刷新后可以成功。其他功能使用正常。",
  "media": [],
  "video_frames": 16,
  "questions": {
    "decision": {
      "type": "choice",
      "instructions": "根据用户反馈，选择最合适的处理优先级。",
      "require_calibrated": false,
      "criteria": {
        "high": "立即处理，关键功能完全不可用",
        "medium": "尽快修复，功能受影响但有替代方式",
        "low": "排期改进，不影响主要功能"
      }
    }
  }
}'
