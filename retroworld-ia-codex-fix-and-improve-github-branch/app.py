from src.retroworld_ia import create_app
from src.retroworld_ia import config


app = create_app()


if __name__ == "__main__":
    port = int(config._env("PORT", "5000"))
    if config.should_use_gunicorn():
        cmd = config.gunicorn_cmd(port)
        print("[BOOT] mode gunicorn auto:", " ".join(cmd), flush=True)
        import os
        os.execvp(cmd[0], cmd)
    print("[BOOT] mode flask dev", flush=True)
    app.run(host="0.0.0.0", port=port, debug=config.DEBUG_LOGS)
