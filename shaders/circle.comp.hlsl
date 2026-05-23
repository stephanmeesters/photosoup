RWTexture2D<uint> target_image : register(u0, space0);

[numthreads(16, 16, 1)]
void main(uint3 global_id : SV_DispatchThreadID) {
    uint width;
    uint height;
    target_image.GetDimensions(width, height);

    uint2 pixel = global_id.xy;
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    float2 size = float2(width, height);
    float2 uv = (float2(pixel) + float2(0.5, 0.5)) / size;
    float2 centered = uv * 2.0 - 1.0;
    centered.x *= size.x / size.y;

    float distance_from_center = length(centered);
    float radius = 0.42;
    float outline_width = 0.025;
    float edge = 2.0 / min(size.x, size.y);

    float fill = 1.0 - smoothstep(radius - edge, radius + edge, distance_from_center);
    float outline_outer = 1.0 - smoothstep(
        radius + outline_width - edge,
        radius + outline_width + edge,
        distance_from_center
    );
    float outline_inner = 1.0 - smoothstep(radius - edge, radius + edge, distance_from_center);
    float outline = saturate(outline_outer - outline_inner);

    float3 background = float3(0.035, 0.045, 0.075);
    float3 circle = float3(0.10, 0.55, 0.85);
    float3 stroke = float3(0.98, 0.82, 0.26);

    float3 color = lerp(background, circle, fill);
    color = lerp(color, stroke, outline);

    uint3 color_bytes = uint3(saturate(color) * 255.0 + 0.5);
    target_image[pixel] =
        color_bytes.r |
        (color_bytes.g << 8) |
        (color_bytes.b << 16) |
        (255u << 24);
}
