from fastapi import FastAPI, HTTPException


app = FastAPI()
items:list = []

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/items")
async def create_item(item: str) -> list[str]:
    items.append(item)
    return items


@app.get("/items/{item_id}")
async def get_item(item_id: int) -> str:
    if item_id >= len(items):
        raise HTTPException(status_code=404, detail="Index out of range")
    else:
        return items[item_id]
 