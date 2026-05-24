import os, json, datetime, shutil, glob

class Session:
    def __init__(self, drive_path, session_file="session_state.json"):
        self.drive_path = drive_path
        self.session_path = os.path.join(drive_path, session_file)
        os.makedirs(drive_path, exist_ok=True)

    def load(self):
        if os.path.exists(self.session_path):
            with open(self.session_path) as f:
                return json.load(f)
        return {"steps": {}, "params": {}, "created_at": None, "updated_at": None}

    def save(self, session):
        session["updated_at"] = datetime.datetime.now().isoformat()
        with open(self.session_path, "w") as f:
            json.dump(session, f, indent=2)

    def mark_step(self, step_name, session=None):
        if session is None:
            session = self.load()
        session["steps"][step_name] = True
        self.save(session)

    def is_step_done(self, step_name, session=None):
        if session is None:
            session = self.load()
        return session["steps"].get(step_name, False)

    def set_param(self, key, value, session=None):
        if session is None:
            session = self.load()
        session["params"][key] = value
        self.save(session)

    def get_param(self, key, default=None, session=None):
        if session is None:
            session = self.load()
        return session["params"].get(key, default)

    def reset(self):
        if os.path.exists(self.session_path):
            os.remove(self.session_path)
        session = {
            "steps": {},
            "params": {},
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": None,
        }
        self.save(session)
        return session

    def checkpoint_path(self, name):
        return os.path.join(self.drive_path, name)

    def save_to_drive(self, src, name):
        dst = self.checkpoint_path(name)
        if os.path.exists(src):
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def restore_from_drive(self, name, dst):
        src = self.checkpoint_path(name)
        if os.path.exists(src):
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            return True
        return False

    def checkpoint_exists(self, name):
        return os.path.exists(self.checkpoint_path(name))

    def print_status(self, session=None):
        if session is None:
            session = self.load()
        print("=" * 60)
        if session["created_at"]:
            print(f"  Session created: {session['created_at'][:19]}")
            done = [k for k, v in session["steps"].items() if v]
            print(f"  Completed steps: {len(done)}")
            for s in done:
                print(f"    - {s}")
            params = session.get("params", {})
            if params:
                print(f"  Params: {json.dumps(params, default=str)}")
        else:
            print("  No previous session.")
        print("=" * 60)


def get_drive_path():
    DRIVE_PATH = "/content/drive/MyDrive/arena_3dgs"
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_PATH, exist_ok=True)
    return DRIVE_PATH
