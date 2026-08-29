from PIL import Image, ImageDraw, ImageFont
import typing
from pathlib import Path

def draw_grounding_boxes(
    image_path: typing.Union[str, Path], 
    boxes: typing.List[typing.List[float]], 
    labels: typing.List[str] = None,
    output_path: typing.Optional[typing.Union[str, Path]] = None
) -> Image.Image:
    """
    Draws normalized bounding boxes [xmin, ymin, xmax, ymax] on an image.
    Outputs a PIL Image, and optionally saves it.
    
    This is extremely useful for generating visual proof for the hackathon judges.
    """
    image = Image.open(image_path).convert("RGBA")
    
    # Create an overlay for drawing transparent elements
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size

    for idx, box in enumerate(boxes):
        if len(box) != 4:
            continue
            
        xmin, ymin, xmax, ymax = box
        
        # Denormalize coordinates [0, 1] to absolute pixel coordinates
        abs_xmin = int(xmin * width)
        abs_ymin = int(ymin * height)
        abs_xmax = int(xmax * width)
        abs_ymax = int(ymax * height)

        # Draw a beautiful neon green box
        box_color = (0, 255, 0, 200) # Neon Green with slight transparency
        draw.rectangle(
            [(abs_xmin, abs_ymin), (abs_xmax, abs_ymax)], 
            outline=box_color, 
            width=3
        )
        
        # Add a text label above the box if provided
        if labels and idx < len(labels):
            label = labels[idx]
            # Draw a subtle background for text readability
            text_width = len(label) * 7 + 10
            draw.rectangle(
                [(abs_xmin, max(0, abs_ymin - 20)), (abs_xmin + text_width, abs_ymin)], 
                fill=box_color
            )
            draw.text((abs_xmin + 5, max(2, abs_ymin - 18)), label, fill=(0, 0, 0, 255))

    # Composite the overlay onto the original image
    result = Image.alpha_composite(image, overlay).convert("RGB")
    
    if output_path:
        result.save(output_path)
        
    return result
