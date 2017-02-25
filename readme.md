# 简单的垃圾消息检测器

使用逻辑回归做二元分类。可作为一个比关键字检测泛化更好的检测器。


```python
from spam import SpamDetector
```


```python
from base import do
```


```python
spams = do('./data/spams.txt',
           open,
           lambda f: f.read().splitlines(),
           lambda lines: map(lambda l: (l, 1), lines)
          )
```


```python
nonspams = do('./data/nonspams.txt',
              open,
              lambda f: f.read().splitlines(),
              lambda lines: map(lambda l: (l, 0), lines)
             )
```


```python
train_set = spams + nonspams
```


```python
SpamDetector.train(train_set)
```


```python
recall_test_set = [
#spams
"100元=15000钻石送VIP10+月卡*3，PK闯关无压力Q504176767",
"招ios的兄弟备战新区：（463130109），送手冲，有礼 包，教攻略",
"100=10000钻石+月卡3*志送VIP10，PK闯关无压力Q209060188",
"英雄联盟D1砖石2D6大师，谁共享账号我带他。wx838557868",
"100元=10000钻石+月卡3张+直送vip10，PK闯关无压力Q209060188 先",
"招ios的兄弟一起去新区Q群：（463130109），送手冲，有礼包，教",
"九九=10000zuan石⋯越卡3張⋯送會員10⋯扣⋯209060188 備戰",
"[哔~] vip8  200支付宝 要的微信加 h10653605",
"100=10000钻石+月卡3*志送VIP10，PK闯关无压力Q209060188100=100",
"100=10000钻+月卡3*直到VIP10，M209060188",
"史诗男爵100=2万钻送V9+英雄萝莉+橙色装备无尽加Q482381396货到付款",
"打小就牛鼻[哔~]币收大V号13904599369",
# nonspams
"萝莉好还是女枪好啊？",
"嗯？",
"打了一早上",
"英雄联盟D1砖石2D6大师，谁共享账号我带他。",
"啦啦啦",
"dddddwsd",
"没人",
"没人啦？",
"……到V11要多少钱？",
"什么删档内测？都公测了",
"冲了月卡怎么领不了",
"刷刷刷！",
"..",
"召唤英灵，V11的",
"没人玩的吗。？",
"钻石干嘛的这上面？",
"半夜不睡+1",
"喵小雨10区交流群180688185 欢迎各位^_^",
"迈巴赫超跑俱乐部：大量v11 霸服3级家族 诚邀v9以上 或38级活跃玩家！",
"德诺之王 公会 欢迎各路大神倾情加盟！微信群：245346603！",
]
```


```python
print "\n".join(map(lambda l: "{1} {0}".format(l, SpamDetector.get_score(l)), recall_test_set))
```

    3.042585997 100元=15000钻石送VIP10+月卡*3，PK闯关无压力Q504176767
    4.57836810458 招ios的兄弟备战新区：（463130109），送手冲，有礼 包，教攻略
    5.45370953702 100=10000钻石+月卡3*志送VIP10，PK闯关无压力Q209060188
    -0.330129462359 英雄联盟D1砖石2D6大师，谁共享账号我带他。wx838557868
    5.10770689611 100元=10000钻石+月卡3张+直送vip10，PK闯关无压力Q209060188 先
    3.78023668478 招ios的兄弟一起去新区Q群：（463130109），送手冲，有礼包，教
    3.69555283132 九九=10000zuan石⋯越卡3張⋯送會員10⋯扣⋯209060188 備戰
    -0.715565761267 [哔~] vip8  200支付宝 要的微信加 h10653605
    8.38936769248 100=10000钻石+月卡3*志送VIP10，PK闯关无压力Q209060188100=100
    2.75154042967 100=10000钻+月卡3*直到VIP10，M209060188
    4.51573459645 史诗男爵100=2万钻送V9+英雄萝莉+橙色装备无尽加Q482381396货到付款
    1.01245636273 打小就牛鼻[哔~]币收大V号13904599369
    -9.60975538763 萝莉好还是女枪好啊？
    -7.51070694457 嗯？
    -7.90653723662 打了一早上
    -4.00656444008 英雄联盟D1砖石2D6大师，谁共享账号我带他。
    -6.71607643485 啦啦啦
    -8.47267000528 dddddwsd
    -7.68230753541 没人
    -8.24183167886 没人啦？
    -6.85830303246 ……到V11要多少钱？
    -8.22823924019 什么删档内测？都公测了
    -9.98193393763 冲了月卡怎么领不了
    -7.57658177589 刷刷刷！
    -7.33409079039 ..
    -6.40476511461 召唤英灵，V11的
    -9.11048468484 没人玩的吗。？
    -8.38023589522 钻石干嘛的这上面？
    -7.31100619677 半夜不睡+1
    -1.39242523214 喵小雨10区交流群180688185 欢迎各位^_^
    -4.74588940525 迈巴赫超跑俱乐部：大量v11 霸服3级家族 诚邀v9以上 或38级活跃玩家！
    -3.22232249994 德诺之王 公会 欢迎各路大神倾情加盟！微信群：245346603！

