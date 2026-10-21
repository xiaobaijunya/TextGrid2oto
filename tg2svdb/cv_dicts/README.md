# C+V 字典（给「添加ph_num」用）

每个语言一个文件夹，内含：

```
<语言>/
├── vowels.txt       # 元音（组起点）
└── consonants.txt   # 辅音（跟在元音后，归入同一组）
```

文件格式：**空白分隔**（空格或换行均可），例如：

```
a i u e o N pau sil br
```

用法：

- **GUI**：`5.标记生成 → transcriptions.csv制作 → 添加ph_num`，
  「规则来源」选 **元音 + 辅音文件**，元音选 `vowels.txt`，辅音选 `consonants.txt`；
- **命令行**：

  ```bash
  python build_diffdataset/add_ph_num.py transcriptions.csv \
      --vowels tg2svdb/cv_dicts/mandarin-xsampa/vowels.txt \
      --consonants tg2svdb/cv_dicts/mandarin-xsampa/consonants.txt
  ```

规则：每个元音 + 其后连续的辅音算一组，`ph_num` 记录每组音素个数。
所以 `pau / sil / br` 放在元音里（各自独立成组），`cl`（促音/长音标记）放在辅音里
（跟在前一个元音后）。

## 各语言的划分

| 语言 | 元音 | 辅音 |
| --- | --- | --- |
| `japanese-romaji` | a i u e o N pau sil br | cl t d s sh j z ts k kw g gw h b p f ch ry ky py dy ty ny hy my gy by n m r w v y |
| `mandarin-xsampa` | a A o @ e 7 U u i i\ i` y AU @U ia iA iAU ie iE iU i@U y{ yE ua uA u@ ue uo :\i r\` :n N pau sil br | p ph t th k kh ts\ ts tsh ts` ts`h x f s s` ts\h s\ m n l z` w j cl |
| `english-arpabet` | aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw pau sil br | b ch d dx dr dw dh f g hh jh k l m n ng p q r s sh t tr tw th v w y z zh cl |
| `cantonese-xsampa` | a 6 E e i I O o u U 9 8 y m= N= :i :u pau sil br | ts tsh f h s l m n N w j p ph t th k kh kw kwh :m :n :N :p_} :t_} :k_} cl |
| `spanish-xsampa` | a e i o u U I y pau sil br | ll b B d D g k p t l rr r m n N J f s C sh ch x cl |

说明：

- 日语 `N`（拨音）单独成音节 → 元音；`ry/ky/py/...` 是腭化声母 → 辅音；
- 粤语 `:m :n :N :p_} :t_} :k_}` 是韵尾 → 辅音，`m= N=` 是成音节鼻音 → 元音；
- 英语 `er` 是 r 化元音 → 元音，`y` 是滑音 → 辅音；
- 西班牙语 `y` 作元音、`ll` 作辅音。
