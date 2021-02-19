## 이미지를 spirng boot에 저장하고 front에서 불러올때

```
@GetMapping(
  value = "/get-image-with-media-type",
  produces = MediaType.IMAGE_JPEG_VALUE
)
public @ResponseBody byte[] getImageWithMediaType() throws IOException {
    InputStream in = getClass().getResourceAsStream("/com/baeldung/produceimage/image.jpg");
    return IOUtils.toByteArray(in);
}
```
