/*
 * @Author: LetMeFly
 * @Date: 2024-09-02 11:39:20
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-09-02 12:06:32
 */
// 培养 -> 课程和环节完成查询
tbody = document.querySelector("#contentParent_dgData > tbody")
trs =  [].slice.apply(tbody.querySelectorAll('tr'))
trs.shift()
for (let i = 0; i < trs.length; i++) {
    tr = trs[i]
    ths = tr.querySelectorAll('td')
    ths[3].innerText = '123456' + `${i + 1}`.padStart(2, '0')
    ths[3].innerText = `课程名${i + 1}`
    ths[3].innerText = 100
}
// 然后发现表格每一行td个数不一样，算了