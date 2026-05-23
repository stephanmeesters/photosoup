struct VsInput {
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(1)]] float4 color : COLOR0;
};

struct VsOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
};

VsOutput main(VsInput input) {
    VsOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}
