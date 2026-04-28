#!/bin/bash

PACKAGE_NAME="com.example.curl_piseps"
REMOTE_PATH="/sdcard/Android/data/$PACKAGE_NAME/files/data"
LOCAL_PATH="./data"

ADB_CMD="adb"

if ! command -v adb &> /dev/null; then
    if [ -f "android/local.properties" ]; then
        SDK_DIR=$(grep "^sdk.dir=" android/local.properties | cut -d'=' -f2)
        if [ -n "$SDK_DIR" ] && [ -f "$SDK_DIR/platform-tools/adb" ]; then
            ADB_CMD="$SDK_DIR/platform-tools/adb"
        fi
    fi
fi

if ! command -v "$ADB_CMD" &> /dev/null && [ "$ADB_CMD" == "adb" ]; then
    echo "Error: adb not found. Please install adb or add it to your PATH."
    echo "Common command: sudo apt install adb"
    exit 1
fi

echo "Using adb: $ADB_CMD"
echo "Pulling data from $REMOTE_PATH to $LOCAL_PATH..."

mkdir -p "$LOCAL_PATH/perfect"
mkdir -p "$LOCAL_PATH/imperfect"

# Pull data
"$ADB_CMD" pull "$REMOTE_PATH" "$LOCAL_PATH/.."

echo "Data pull complete."
echo "New samples in $LOCAL_PATH:"
ls -R "$LOCAL_PATH" | grep ":"
