"""
Tool implementations called by the agent.
Each function returns a plain dict that gets serialised back to the LLM.
"""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
from assistant.retrieval import get_by_sku, get_all, search


# ── Pydantic schema for create_order ────────────────────────────────────────

class OrderLineItem(BaseModel):
    sku: str
    quantity: int = Field(gt=0)


class OrderRequest(BaseModel):
    dealer_name: str
    line_items: List[OrderLineItem]


class OrderConfirmation(BaseModel):
    order_id: str
    dealer_name: str
    line_items: List[dict]
    total_inr: int
    status: str
    placed_at: str


# ── Tool functions ───────────────────────────────────────────────────────────

def check_stock(sku: str) -> dict:
    """Look up availability for a product by SKU."""
    product = get_by_sku(sku)
    if product is None:
        return {"error": f"SKU '{sku}' not found in catalogue."}
    return {
        "sku": product["sku"],
        "name": product["name"],
        "stock": product["stock"],
        "price_inr": product["price_inr"],
        "available": product["stock"] > 0,
    }


def find_parts_by_vehicle(make_model: str) -> dict:
    """Find all parts that fit a given vehicle (make + model string)."""
    all_parts = get_all()
    query = make_model.lower().strip()

    # Exact / substring match on vehicle_fitment
    matches = [
        p for p in all_parts
        if query in p.get("vehicle_fitment", "").lower()
        or p.get("vehicle_fitment", "").lower() in query
    ]

    # If few exact matches, fall back to semantic search
    if len(matches) < 3:
        semantic = search(make_model, n_results=10)
        seen_skus = {m["sku"] for m in matches}
        for s in semantic:
            if s["sku"] not in seen_skus:
                matches.append(s)
                seen_skus.add(s["sku"])

    if not matches:
        return {"error": f"No parts found for vehicle: {make_model}"}

    return {
        "vehicle": make_model,
        "count": len(matches),
        "parts": [
            {
                "sku": p["sku"],
                "name": p["name"],
                "category": p["category"],
                "price_inr": p["price_inr"],
                "stock": p["stock"],
            }
            for p in matches[:15]  # cap at 15 results
        ],
    }


def create_order(dealer_name: str, line_items: list[dict]) -> dict:
    """
    Place an order. line_items is a list of {"sku": str, "quantity": int}.
    Returns a structured OrderConfirmation or an error dict.
    """
    try:
        req = OrderRequest(
            dealer_name=dealer_name,
            line_items=[OrderLineItem(**item) for item in line_items],
        )
    except Exception as e:
        return {"error": f"Invalid order format: {e}"}

    resolved_items = []
    total = 0
    errors = []

    for item in req.line_items:
        product = get_by_sku(item.sku)
        if product is None:
            errors.append(f"SKU '{item.sku}' not found.")
            continue
        if product["stock"] < item.quantity:
            errors.append(
                f"Insufficient stock for {item.sku} ({product['name']}): "
                f"requested {item.quantity}, available {product['stock']}."
            )
            continue
        line_total = product["price_inr"] * item.quantity
        total += line_total
        resolved_items.append({
            "sku": item.sku,
            "name": product["name"],
            "quantity": item.quantity,
            "unit_price_inr": product["price_inr"],
            "line_total_inr": line_total,
        })

    if errors:
        return {"error": "; ".join(errors)}

    confirmation = OrderConfirmation(
        order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
        dealer_name=req.dealer_name,
        line_items=resolved_items,
        total_inr=total,
        status="confirmed",
        placed_at=datetime.utcnow().isoformat() + "Z",
    )
    return confirmation.model_dump()


# ── Groq / OpenAI-style tool declarations ────────────────────────────────────

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check the current stock and price for a specific product SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Product SKU code, e.g. BRK-1042"},
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_parts_by_vehicle",
            "description": "Find all available parts that fit a specific vehicle make and model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "make_model": {
                        "type": "string",
                        "description": "Vehicle make and model, e.g. 'Bajaj Pulsar 150' or 'KTM Duke 390'",
                    },
                },
                "required": ["make_model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Place an order for a dealer with one or more line items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dealer_name": {"type": "string", "description": "Name of the dealer placing the order"},
                    "line_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer"},
                            },
                            "required": ["sku", "quantity"],
                        },
                        "description": "List of items to order",
                    },
                },
                "required": ["dealer_name", "line_items"],
            },
        },
    },
]

TOOL_MAP = {
    "check_stock": check_stock,
    "find_parts_by_vehicle": find_parts_by_vehicle,
    "create_order": create_order,
}
