import random
import math
from typing import List, Tuple, TypeVar, Any

T = TypeVar('T')  # Generic type for the items

def exp_weighted_choice(items, decay=0.5):
    """Select item from list with exponentially decreasing probability.
    Args:
        items: List of items to choose from
        decay: Exponential decay factor (0-1, smaller means steeper decay)
    Returns:
        Selected item
    """
    if not items:
        return None
    if len(items) == 1:
        return items[0]
        
    # Calculate weights as decay^i
    weights = [decay ** i for i in range(len(items))]
    # Normalize weights to sum to 1
    total = sum(weights)
    weights = [w/total for w in weights]
    
    return random.choices(items, weights=weights, k=1)[0]

def pick_weighted(scored_items: List[Tuple[T, float]], weight: float = 1.0, normalization=100.0, n: int = 1) -> List[T]:
    """Choose up to n items randomly weighted by score without replacement.
    
    Args:
        scored_items: List of (item, score) tuples where scores are 0-100
        weight: Power to raise normalized scores to (default 1.0)
               Higher values increase probability of higher scored items
        normalization: Value to normalize scores against (default 100.0)
        n: Maximum number of items to select (default 1)
               
    Returns:
        List of up to n chosen items (or single item if n=1)
    """
    if not scored_items:
        raise ValueError("Cannot choose from empty sequence")
        
    if n == 0:
        return []
        
    # Limit n to available items
    n = min(n, len(scored_items))
        
    # Make copy to avoid modifying original
    remaining_items = list(scored_items)
    chosen = []
    
    # Keep selecting until we have n items
    while len(chosen) < n:
        # Normalize scores to 0-1 range
        max_score = max(score for _, score in remaining_items)
        if max_score == 0:
            # If all scores are 0, choose randomly
            item = random.choice(remaining_items)[0]
            chosen.append(item)
            remaining_items = [x for x in remaining_items if x[0] is not item]
            continue
            
        # Calculate weighted probabilities
        weighted_scores = [(item, math.pow(score/max(max_score, normalization), weight)) 
                          for item, score in remaining_items]
        
        # Calculate cumulative probabilities
        total = sum(score for _, score in weighted_scores)
        if total == 0:
            item = random.choice(remaining_items)[0]
            chosen.append(item)
            remaining_items = [x for x in remaining_items if x[0] is not item]
            continue
            
        r = random.uniform(0, total)
        cumulative = 0
        selected = False
        for item, score in weighted_scores:
            cumulative += score
            if r <= cumulative:
                chosen.append(item)
                remaining_items = [x for x in remaining_items if x[0] is not item]
                selected = True
                break
                
        # Fallback in case of floating point rounding issues
        if not selected:
            chosen.append(remaining_items[-1][0])
            remaining_items.pop()
            
    return chosen[0] if n == 1 else chosen

if __name__ == "__main__":
    items = [('a', 100), ('b', 50), ('c', 25), ('d', 10)]
    
    # Test single selection
    print("Single selection:", pick_weighted(items, weight=2.0))
    
    # Test multiple selection
    print("Multiple selection:", pick_weighted(items, weight=2.0, n=3))
