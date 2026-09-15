import base64
from pathlib import Path

class MessageItem:
    def __init__(self, value: dict):
        self.value: dict = value

class Image(MessageItem):
    def __init__(self, fp: str, *, b64: bool = False):
        """Create an image ready to be appended to the chat
        - fp: The local path of the image, or image URL.
        - b64: `fp` representing the Base64 value of the image, not a path."""

        if b64:
            path = f"data:image/jpeg;base64,{fp}"

        elif 'http' in fp.lower():
            path = fp
        else:
            self.path: Path | str = Path(fp)
            if not self.path.is_file():
                raise ValueError(f"The file \"{self.path.name}\" not found")
            with open(self.path, "rb") as image_file:
                b64 = base64.b64encode(image_file.read()).decode("utf-8")
                path = f"data:image/jpeg;base64,{b64}"
                image_file.close()

        super().__init__(value={
                    "type": "image_url",
                    "image_url": {
                        "url": path
                    }})