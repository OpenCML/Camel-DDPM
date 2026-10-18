"""Export a balanced, image-deduplicated transport queue on the server."""
import argparse
import json
from pathlib import Path
import sqlite3
from urllib.parse import unquote, urlencode, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    db = sqlite3.connect("file:" + str(args.index.resolve()) + "?mode=ro", uri=True)
    cursors = [iter(db.execute("SELECT image_id, record FROM pairs WHERE language=? ORDER BY identity", (language,))) for language in ("zh", "en")]
    seen, queue = set(), []
    while len(queue) < args.limit and cursors:
        for cursor in cursors[:]:
            try:
                image_id, record = next(cursor)
            except StopIteration:
                cursors.remove(cursor)
                continue
            if image_id in seen:
                continue
            row = json.loads(record)
            path = urlsplit(row["image_url"]).path
            title = unquote(path.split("/")[-2 if path.startswith("/wikipedia/commons/thumb/") else -1])
            api = "https://commons.wikimedia.org/w/api.php?" + urlencode({"action": "query", "format": "json", "prop": "imageinfo",
                "iiprop": "url|size|sha1|mime|extmetadata", "iiurlwidth": 768, "iiurlheight": 768, "titles": "File:" + title})
            captions = [json.loads(value) for (value,) in db.execute("SELECT record FROM pairs WHERE image_id=? ORDER BY identity", (image_id,))]
            queue.append({"image_id": image_id, "image_url": row["image_url"], "api_url": api, "captions": captions})
            seen.add(image_id)
            if len(queue) == args.limit:
                break
    db.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(queue, stream, ensure_ascii=False)
    print(json.dumps({"queue_images": len(queue), "caption_pairs_by_language": {
        language: sum(pair["language"] == language for item in queue for pair in item["captions"]) for language in ("en", "zh")}}))


if __name__ == "__main__":
    main()
