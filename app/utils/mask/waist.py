from utils.mask.body_common import validate_body, create_body_mask, adjust_body_part

def validate_waist(image_matrix):
    return validate_body(image_matrix)

def create_waist_mask(image_matrix):
    return create_body_mask(image_matrix)

def waist_adjust(image_matrix, body_cache_data, scale_val=0.0):
    return adjust_body_part(image_matrix, body_cache_data, scale_val=scale_val, t_center=0.55, sigma=0.09)